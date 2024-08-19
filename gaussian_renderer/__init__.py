import numpy as np
import jittor as jt
import math
from gaussian_renderer.diff_gaussian_rasterizater import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.mesh_based_gaussian_model import MeshBasedGaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import strip_symmetric, build_scaling_rotation


def build_covariance_from_scaling_rotation(scaling, rotation):
            L = build_scaling_rotation(scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance

def convert_conv(x):
    conv3D = jt.zeros((x.shape[0],6))
    conv3D[:,0] = x[:,0,0]
    conv3D[:,1] = x[:,0,1]
    conv3D[:,2] = x[:,0,2]
    conv3D[:,3] = x[:,1,1]
    conv3D[:,4] = x[:,1,2]
    conv3D[:,5] = x[:,2,2]
    return conv3D

def render(viewpoint_camera, pc : MeshBasedGaussianModel, pipe, bg_color : jt.Var, scaling_modifier = 1.0, override_color = None,bg_gaussian=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points1 = pc.screenspace_points
    if(bg_gaussian!=None):
        screenspace_points2 = jt.zeros_like(bg_gaussian.get_xyz, dtype=bg_gaussian.get_xyz.dtype) + 0
        screenspace_points = jt.concat([screenspace_points1,screenspace_points2],dim=0)
    else:
        screenspace_points = screenspace_points1
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    vertex1 = pc.vertex1
    vertex2 = pc.vertex2
    vertex3 = pc.vertex3
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = jt.clamp(sh2rgb + 0.5, min_v=0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if(bg_gaussian != None):
        bg_scale = bg_gaussian.get_scaling
        bg_rot = bg_gaussian.get_rotation
        bg_cov3D = build_covariance_from_scaling_rotation(bg_scale, bg_rot)
        bg_cov3D = convert_conv(bg_cov3D)
        bg_mean3D = bg_gaussian.get_xyz
        bg_shs = bg_gaussian.get_features
        bg_opacity = bg_gaussian.get_opacity

        shs_view = bg_shs.transpose(1, 2).view(-1, 3, (3+1)**2)
        dir_pp = (bg_mean3D - viewpoint_camera.camera_center.repeat(bg_shs.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        bg_sh2rgb = eval_sh(3, shs_view, dir_pp_normalized)
        bg_colors_precomp = jt.clamp_min(bg_sh2rgb + 0.5, 0.0)
        
        means3D = jt.concat([means3D, bg_mean3D], dim=0)
        opacity =jt.concat([opacity, bg_opacity], dim=0)
        cov3D_precomp = jt.concat([cov3D_precomp, bg_cov3D], dim=0)
        if(shs!=None):
            shs = jt.concat([shs, bg_shs], dim=0)
        else:
            colors_precomp = jt.concat([colors_precomp,bg_colors_precomp], dim=0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "vertex1":vertex1,
            "vertex2":vertex2,
            "vertex3":vertex3,
            "scale":scales}


def bg_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : jt.Var, scaling_modifier = 1.0, override_color = None,mesh_gaussians=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # num = pc.get_xyz.shape[0] + mesh_gaussians.get_bc.shape[0]
    # screenspace_points = jt.zeros((num,3), dtype=pc.get_xyz.dtype) + 0
    screenspace_points1 = pc.screenspace_points
    if(mesh_gaussians!=None):
        screenspace_points2 = mesh_gaussians.screenspace_points
        screenspace_points = jt.concat([screenspace_points1,screenspace_points2],dim=0)
    # else:
    #     screenspace_points = screenspace_points1
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:

        scales = pc.get_scaling
        rotations = pc.get_rotation
        

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = jt.clamp(sh2rgb + 0.5, min_v=0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if(mesh_gaussians != None):
        mesh_scale = mesh_gaussians.get_scaling.stop_grad()
        mesh_rot = mesh_gaussians.get_rotation.stop_grad()
        mesh_mean3D = mesh_gaussians.get_xyz.stop_grad()
        mesh_shs = mesh_gaussians.get_features.stop_grad()
        mesh_opacity = mesh_gaussians.get_opacity.stop_grad()

        means3D = jt.concat([means3D, mesh_mean3D], dim=0)
        scales = jt.concat([scales, mesh_scale], dim=0)
        rotations = jt.concat([rotations, mesh_rot], dim=0)
        shs = jt.concat([shs, mesh_shs], dim=0)
        opacity = jt.concat([opacity, mesh_opacity], dim=0)
    else:
        mean3D = means3D
        scale = scales
        rotation = rotations
        sh = shs
        o = opacity
    # print(means3D[0])
    # print(means2D.shape)
    # print(opacity.shape)
    # print(rotation.shape)
    # print(scale.shape)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
