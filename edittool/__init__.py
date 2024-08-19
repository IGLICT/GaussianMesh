import numpy as np
import jittor as jt
from jittor import nn
import os
from os import makedirs
import time
import tqdm
import igl
import copy
import numpy as np
import math
import json
from gaussian_renderer.diff_gaussian_rasterizater import GaussianRasterizationSettings, NewGaussianRasterizer
from edittool.pose_utils import generate_ellipse_path, generate_spherical_sample_path, generate_spiral_path, generate_spherify_path,  gaussian_poses, circular_poses,getWorld2View2
from edittool.bg_gaussian import BGGaussianModel
from edittool.mesh_based_gaussian import MeshBasedGaussianModel
from edittool.general_utils import strip_symmetric, get_barycentric_coordinate
from edittool.sh_utils import eval_sh
from edittool.camera_utils import GSCamera
from edittool.graphics_utils import focal2fov
import pyACAP

def rotation_matrix_to_quaternion(rotation_matrix):
    #return :(N,4)
    r00 = rotation_matrix[:, 0, 0]
    r01 = rotation_matrix[:, 0, 1]
    r02 = rotation_matrix[:, 0, 2]
    r10 = rotation_matrix[:, 1, 0]
    r11 = rotation_matrix[:, 1, 1]
    r12 = rotation_matrix[:, 1, 2]
    r20 = rotation_matrix[:, 2, 0]
    r21 = rotation_matrix[:, 2, 1]
    r22 = rotation_matrix[:, 2, 2]
    w = jt.sqrt(1 + r00 + r11 + r22) / 2
    x = (r21 - r12) / ((4 * w))
    y = (r02 - r20) / ((4 * w))
    z = (r10 - r01) / ((4 * w))
    return (jt.stack([w,x,y,z],dim=1)).normalize()

class SingleObjectDeform:
    def __init__(self, fg_path,mesh_path,name = None):
        self.load_gaussian(fg_path)
        self.load_mesh(mesh_path)
        self.name = name
        if(name== None):
            self.name = mesh_path
    def get_name(self):
        return self.name
    def load_gaussian(self,gaussian_path):
        self.gaussians = MeshBasedGaussianModel(3) 
        self.gaussians.load_ply(gaussian_path)
        self.gaussian_pos = self.gaussians.get_load_xyz
        self.gaussian_proj_pos  = self.gaussians.get_proj_xyz
        self.gaussian_cov = self.gaussians.get_covariance()
        self.gaussian_o = self.gaussians.get_opacity
        self.gaussian_feature = self.gaussians.get_features
        self.gaussian_deform_cov = self.gaussian_cov
        self.gaussian_deform_pos = self.gaussian_pos
        self.number_gaussian = self.gaussians.get_load_xyz.shape[0]
        self.deform_rot = jt.init.eye(3).unsqueeze(0).expand(self.gaussian_o.shape[0], -1, -1).float().cuda()
        try:
            self.index_tri = self.gaussians.fid
        except:
            self.index_tri = None
    def load_mesh(self,mesh_path):
        vertex, triangles = igl.read_triangle_mesh(mesh_path)

        if(self.index_tri is None):
            normals = np.cross(vertex[triangles[:, 1]] - vertex[triangles[:, 0]], vertex[triangles[:, 2]] - vertex[triangles[:, 0]])
            normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
            index1 = triangles[:,0]
            v_on_triangles1 = vertex[index1]
            index2 = triangles[:,1]
            v_on_triangles2 = vertex[index2]
            index3 = triangles[:,2]
            v_on_triangles3 = vertex[index3]
            bias = -(v_on_triangles1*normals).sum(axis = 1) #[num_tri]
            gaussian_pos_numpy = self.gaussian_pos.numpy()
            gaussian_proj_pos_numpy = self.gaussian_proj_pos.numpy()
            sqrD,index_tri,closest_p=igl.point_mesh_squared_distance(gaussian_proj_pos_numpy, vertex, triangles)
            n_gaussian = normals[index_tri]
            b_gaussian = bias[index_tri]
            distance = -((n_gaussian*gaussian_pos_numpy).sum(axis=1)+b_gaussian) #[num_tri]
            gaussian_intersection = gaussian_pos_numpy+np.expand_dims(distance,axis=1)*n_gaussian
            self.index_tri = index_tri
        else:
            self.gaussian_triangles = triangles[self.index_tri.squeeze(1)]
            # print(self.index_tri.shape)
            # print(gaussian_triangles.shape)
            gaussian_intersection = self.gaussian_proj_pos

        gaussian_triangles_p1 = vertex[self.gaussian_triangles[:,0]]
        gaussian_triangles_p2 = vertex[self.gaussian_triangles[:,1]]
        gaussian_triangles_p3 = vertex[self.gaussian_triangles[:,2]]

        self.coord = get_barycentric_coordinate(gaussian_intersection,gaussian_triangles_p1,
                                                gaussian_triangles_p2,gaussian_triangles_p3)

        self.weight_g_pos = jt.array(np.expand_dims(self.coord, axis=2))
        self.weight_g_rs = jt.array(np.expand_dims(self.coord, axis=(2, 3)))
        self.vertex = jt.array(vertex)
        self.ACAPtool = pyACAP.pyACAP(mesh_path) 
    def deform_gaussian(self, deform_mesh_path):
        # cur_pos:new vertices on deformed mesh [mesh_v,3]
        # cur_rot:deform gradient [mesh_v,3,3]
        # cur_shear:deform gradient [mesh_v,3,3]
        deform_vertex, triangles = igl.read_triangle_mesh(deform_mesh_path)
        
        R1, S1 = self.ACAPtool.GetRS(self.vertex, deform_vertex, 1,  os.cpu_count()//2)

        cur_pos = jt.array(deform_vertex)
        cur_rot = jt.array(R1.reshape((-1, 3, 3)))
        cur_shear = jt.array(S1.reshape((-1, 3, 3)))

        # get deform pos
        delta_pos_ = (cur_pos - self.vertex)[self.gaussian_triangles] 
        g_delta_pos = jt.sum(self.weight_g_pos * delta_pos_, dim=1)
        
        # get deform cov
        R_ = cur_rot[self.gaussian_triangles]
        g_delta_r = jt.sum(self.weight_g_rs * R_, dim=1)
        self.gaussian_deform_rot = g_delta_r.transpose(1,2)

        S_ = cur_shear[self.gaussian_triangles]
        g_delta_s = jt.sum(self.weight_g_rs * S_, dim=1)

        g_delta_rs = jt.matmul(self.gaussian_deform_rot, g_delta_s)

        self.gaussian_deform_cov = jt.matmul(jt.matmul(g_delta_rs, self.gaussian_cov), g_delta_rs.transpose(1,2))
        # self.gaussian_deform_cov = self.gaussian_cov
        self.gaussian_deform_pos = self.gaussian_pos + g_delta_pos      

class SceneVisualTool:
    def __init__(self, bg_gaussian_path = None):
        self.deform_rot = None
        self.load_bg_gaussian(bg_gaussian_path)
        self.gaussians_list = []
    def load_bg_gaussian(self,path):
        bg_gaussian = BGGaussianModel(3)
        bg_gaussian.load_ply(path)
        self.bg_scale = bg_gaussian.get_scaling
        self.bg_rot = bg_gaussian.get_rotation
        self.bg_cov3D = bg_gaussian.get_covariance()
        self.bg_mean3D = bg_gaussian.get_xyz
        self.bg_shs = bg_gaussian.get_features
        self.bg_opacity = bg_gaussian.get_opacity
        self.bg_deform_rot = jt.init.eye(3).unsqueeze(0).repeat(self.bg_scale.shape[0], 1, 1).float()
    def add_gaussian(self,gaussian_path,mesh_path,name = None):
        mesh_gaussian = SingleObjectDeform(gaussian_path,mesh_path,name)
        self.gaussians_list.append(mesh_gaussian)
    def deform_one_gaussian(self,name,deform_mesh_path):
        # matrix_pos(numpy):new vertices on deformed mesh [mesh_v,3]
        # matrix_rot(numpy):deform gradient [mesh_v,3,3]
        # matrix_shear(numpy):deform gradient [mesh_v,3,3]
        for gaussian in self.gaussians_list:
            if(gaussian.get_name()==name):
                gaussian.deform_gaussian(deform_mesh_path)
    def render_gaussian(self,viewpoint_camera):
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=jt.array([1,1,1]).float(),
            scale_modifier=1,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=3,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = NewGaussianRasterizer(raster_settings=raster_settings)

        # gather all gaussians
        means3D = self.bg_mean3D
        shs = self.bg_shs
        deform_rot = self.bg_deform_rot
        cov3D_precomp = self.bg_cov3D
        opacity = self.bg_opacity
        for meshgaussian in self.gaussians_list:
            means3D = jt.concat([means3D, meshgaussian.gaussian_deform_pos], dim=0)
            shs = jt.concat([shs, meshgaussian.gaussian_feature], dim=0)
            deform_rot = jt.concat([deform_rot, meshgaussian.gaussian_deform_rot], dim=0)
            cov3D_precomp = jt.concat([cov3D_precomp, meshgaussian.gaussian_deform_cov], dim=0)
            opacity = jt.concat([opacity, meshgaussian.gaussian_o], dim=0)
        # print("mean3D:",means3D.shape)

        # render begin
        screenspace_points = jt.zeros_like(means3D, dtype=means3D.dtype) + 0
        means2D = screenspace_points

        shs_view = shs.transpose(1, 2).view(-1, 3, (3+1)**2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(shs.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        dir_pp_normalized_rot = jt.matmul(deform_rot.transpose(1,2),dir_pp_normalized.unsqueeze(2)).squeeze(2)
        # print(dir_pp_normalized_rot.shape)
        sh2rgb = eval_sh(3, shs_view, dir_pp_normalized_rot)
        colors_precomp = jt.clamp(sh2rgb + 0.5, min_v=0.0)
 
        eigenvalues, eigenvectors = jt.linalg.eigh(cov3D_precomp)
        eigenvectors = eigenvectors*(nn.sign(jt.array(np.linalg.det(eigenvectors.numpy()))).unsqueeze(1).unsqueeze(1))
        new_s = jt.sqrt(eigenvalues)
        new_q = rotation_matrix_to_quaternion(eigenvectors)
        


        rendered_image, r = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacity,
            scales = new_s,
            rotations = new_q,
            cov3D_precomp = None)
        # rendered_image, r = rasterizer(
        #     means3D = means3D,
        #     means2D = means2D,
        #     shs = None,
        #     colors_precomp = colors_precomp,
        #     opacities = opacity,
        #     scales = None,
        #     rotations = None,
        #     cov3D_precomp = strip_symmetric(cov3D_precomp))
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return rendered_image
    def render_gaussian_test(self,path,name,id=1,output=None):
        # Set up rasterization configuration
        transformsfile ="cameras.json"
        model_path = output
        render_path = os.path.join(model_path, name, "deform_{}".format(id), "renders")
        gts_path = os.path.join(model_path, name, "deform_{}".format(id), "gt")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        idx=0
        with open(os.path.join(path, transformsfile)) as json_file:
            camera_transforms = json.load(json_file)
            for idx, frame in enumerate(camera_transforms):
                # if(idx!=0):
                #     continue
                camera_transform = camera_transforms[idx]
                # Extrinsics
                rot = np.array(camera_transform['rotation'])
                pos = np.array(camera_transform['position'])
                W2C = np.zeros((4,4))
                W2C[:3, :3] = rot
                W2C[:3, 3] = pos
                W2C[3,3] = 1
                Rt = np.linalg.inv(W2C)
                T = Rt[:3, 3]
                R = Rt[:3, :3].transpose()
                # Intrinsics
                width = camera_transform['width']
                height = camera_transform['height']
                fy = camera_transform['fy']
                fx = camera_transform['fx']
                fov_y = focal2fov(fy, height)
                fov_x = focal2fov(fx, width)
                viewpoint_camera = GSCamera(
            colmap_id=idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=camera_transform['img_name'], uid=idx,
            image_height=height, image_width=width,)
                rendered_image = self.render_gaussian(viewpoint_camera)
                # print(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(rendered_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))               
    def get_single_camera(self,path,id=1):
        transformsfile = "cameras.json"
        with open(os.path.join(path, transformsfile)) as json_file:
            camera_transforms = json.load(json_file)
        camera_transform = camera_transforms[id]
                # Extrinsics
        rot = np.array(camera_transform['rotation'])
        pos = np.array(camera_transform['position'])
        W2C = np.zeros((4,4))
        W2C[:3, :3] = rot
        W2C[:3, 3] = pos
        W2C[3,3] = 1
        Rt = np.linalg.inv(W2C)
        T = Rt[:3, 3]
        R = Rt[:3, :3].transpose()
                # Intrinsics
        width = camera_transform['width']
        height = camera_transform['height']
        fy = camera_transform['fy']
        fx = camera_transform['fx']
        fov_y = focal2fov(fy, height)
        fov_x = focal2fov(fx, width)
        viewpoint_camera = GSCamera(
            colmap_id=id, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=camera_transform['img_name'], uid=id,
            image_height=height, image_width=width,)
        return viewpoint_camera
    def get_camera(self,path):
        transformsfile ="cameras.json"
        camera_list = []
        circle_cam_list = []
        with open(os.path.join(path, transformsfile)) as json_file:

            camera_transforms = json.load(json_file)
            for idx, frame in enumerate(camera_transforms):
                # if(idx!=0):
                #     continue
                camera_transform = camera_transforms[idx]
                # Extrinsics
                rot = np.array(camera_transform['rotation'])
                pos = np.array(camera_transform['position'])
                W2C = np.zeros((4,4))
                W2C[:3, :3] = rot
                W2C[:3, 3] = pos
                W2C[3,3] = 1
                Rt = np.linalg.inv(W2C)
                T = Rt[:3, 3]
                R = Rt[:3, :3].transpose()
                # Intrinsics
                width = camera_transform['width']
                height = camera_transform['height']
                fy = camera_transform['fy']
                fx = camera_transform['fx']
                fov_y = focal2fov(fy, height)
                fov_x = focal2fov(fx, width)

                viewpoint_camera = GSCamera(
            colmap_id=idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=camera_transform['img_name'], uid=idx,
            image_height=height, image_width=width,)
                if(idx==0):
                    view = viewpoint_camera
                camera_list.append(viewpoint_camera)
        return camera_list
    def create_circle_cam(self,path,frames=300):
        transformsfile ="cameras.json"
        camera_list = []
        circle_cam_list = []
        with open(os.path.join(path, transformsfile)) as json_file:
            camera_transforms = json.load(json_file)
            for idx, frame in enumerate(camera_transforms):
                # if(idx!=0):
                #     continue
                camera_transform = camera_transforms[idx]
                # Extrinsics
                rot = np.array(camera_transform['rotation'])
                pos = np.array(camera_transform['position'])
                W2C = np.zeros((4,4))
                W2C[:3, :3] = rot
                W2C[:3, 3] = pos
                W2C[3,3] = 1
                Rt = np.linalg.inv(W2C)
                T = Rt[:3, 3]
                R = Rt[:3, :3].transpose()
                # Intrinsics
                width = camera_transform['width']
                height = camera_transform['height']
                fy = camera_transform['fy']
                fx = camera_transform['fx']
                fov_y = focal2fov(fy, height)
                fov_x = focal2fov(fx, width)

                viewpoint_camera = GSCamera(
            colmap_id=idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=camera_transform['img_name'], uid=idx,
            image_height=height, image_width=width,)
                if(idx==0):
                    view = viewpoint_camera
                camera_list.append(viewpoint_camera)
        circle_pose_list = generate_ellipse_path(camera_list,n_frames=frames)
        
        for idx, pose in enumerate(circle_pose_list):
            view.world_view_transform = jt.array(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view.full_proj_transform = jt.nn.bmm(view.world_view_transform.unsqueeze(0),view.projection_matrix.unsqueeze(0)).squeeze(0)
            view.camera_center = jt.linalg.inv(view.world_view_transform)[3, :3]
            circle_cam_list.append(copy.copy(view))
            # print(view.world_view_transform)
        return circle_cam_list
class ObjectVisualTool:
    def __init__(self):
        self.deform_rot = None
        self.gaussians_list = []
    
    def add_gaussian(self,gaussian_path,mesh_path,name = None):
        mesh_gaussian = SingleObjectDeform(gaussian_path,mesh_path,name)
        self.gaussians_list.append(mesh_gaussian)
    
    def deform_one_gaussian(self,name,deform_mesh_path):
        # matrix_pos(numpy):new vertices on deformed mesh [mesh_v,3]
        # matrix_rot(numpy):deform gradient [mesh_v,3,3]
        # matrix_shear(numpy):deform gradient [mesh_v,3,3]
        for gaussian in self.gaussians_list:
            if(gaussian.get_name()==name):
                gaussian.deform_gaussian(deform_mesh_path)
    
    def render_gaussian(self,viewpoint_camera):

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=jt.array([1,1,1]).float(),
            scale_modifier=1,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=3,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = NewGaussianRasterizer(raster_settings=raster_settings)

        # gather all gaussians
        means3D = self.gaussians_list[0].gaussian_deform_pos
        shs = self.gaussians_list[0].gaussian_feature
        deform_rot = self.gaussians_list[0].gaussian_deform_rot
        cov3D_precomp = self.gaussians_list[0].gaussian_deform_cov
        opacity = self.gaussians_list[0].gaussian_o
        if(len(self.gaussians_list)>1):
            for meshgaussian in self.gaussians_list[1:]:
                means3D = jt.concat([means3D, meshgaussian.gaussian_deform_pos], dim=0)
                shs = jt.concat([shs, meshgaussian.gaussian_feature], dim=0)
                deform_rot = jt.concat([deform_rot, meshgaussian.gaussian_deform_rot], dim=0)
                cov3D_precomp = jt.concat([cov3D_precomp, meshgaussian.gaussian_deform_cov], dim=0)
                opacity = jt.concat([opacity, meshgaussian.gaussian_o], dim=0)
        # print("mean3D:",means3D.shape)

        
        
        # render begin
        screenspace_points = jt.zeros_like(means3D, dtype=means3D.dtype) + 0
        means2D = screenspace_points

        shs_view = shs.transpose(1, 2).view(-1, 3, (3+1)**2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(shs.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        dir_pp_normalized_rot = jt.matmul(deform_rot.transpose(1,2),dir_pp_normalized.unsqueeze(2)).squeeze(2)
        # print(dir_pp_normalized_rot.shape)
        sh2rgb = eval_sh(3, shs_view, dir_pp_normalized_rot)
        colors_precomp = jt.clamp(sh2rgb + 0.5, min_v=0.0)
 
        # eigenvalues, eigenvectors = jt.linalg.eigh(cov3D_precomp)
        # # print(nn.sign(jt.array(np.linalg.det(eigenvectors.numpy()))))
        # eigenvectors = eigenvectors*(nn.sign(jt.array(np.linalg.det(eigenvectors.numpy()))).unsqueeze(1).unsqueeze(1))
        # new_s = jt.sqrt(eigenvalues)
        # new_q = rotation_matrix_to_quaternion(eigenvectors)
        # rendered_image, r = rasterizer(
        #     means3D = means3D,
        #     means2D = means2D,
        #     shs = shs,
        #     colors_precomp = None,
        #     opacities = opacity,
        #     scales = new_s,
        #     rotations = new_q,
        #     cov3D_precomp = None)
        rendered_image, r = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = None,
            rotations = None,
            cov3D_precomp = strip_symmetric(cov3D_precomp))
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return rendered_image

    def render_gaussian_test(self,path,name,id=1,output=None):
        # Set up rasterization configuration
        transformsfile ="cameras.json"
        model_path = output
        render_path = os.path.join(model_path, name, "deform_{}".format(id), "renders")
        gts_path = os.path.join(model_path, name, "deform_{}".format(id), "gt")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        idx=0
        with open(os.path.join(path, transformsfile)) as json_file:
            camera_transforms = json.load(json_file)
            for idx, frame in enumerate(camera_transforms):
                # if(idx!=0):
                #     continue
                camera_transform = camera_transforms[idx]
                # Extrinsics
                rot = np.array(camera_transform['rotation'])
                pos = np.array(camera_transform['position'])
                W2C = np.zeros((4,4))
                W2C[:3, :3] = rot
                W2C[:3, 3] = pos
                W2C[3,3] = 1
                Rt = np.linalg.inv(W2C)
                T = Rt[:3, 3]
                R = Rt[:3, :3].transpose()
                # Intrinsics
                width = camera_transform['width']
                height = camera_transform['height']
                fy = camera_transform['fy']
                fx = camera_transform['fx']
                fov_y = focal2fov(fy, height)
                fov_x = focal2fov(fx, width)
                viewpoint_camera = GSCamera(
            colmap_id=idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=camera_transform['img_name'], uid=idx,
            image_height=height, image_width=width,)
                rendered_image = self.render_gaussian(viewpoint_camera)
                # print(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(rendered_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                
    def get_single_camera(self,path,id=1):
        transformsfile = "cameras.json"
        with open(os.path.join(path, transformsfile)) as json_file:
            camera_transforms = json.load(json_file)
        camera_transform = camera_transforms[id]
                # Extrinsics
        rot = np.array(camera_transform['rotation'])
        pos = np.array(camera_transform['position'])
        W2C = np.zeros((4,4))
        W2C[:3, :3] = rot
        W2C[:3, 3] = pos
        W2C[3,3] = 1
        Rt = np.linalg.inv(W2C)
        T = Rt[:3, 3]
        R = Rt[:3, :3].transpose()
                # Intrinsics
        width = camera_transform['width']
        height = camera_transform['height']
        fy = camera_transform['fy']
        fx = camera_transform['fx']
        fov_y = focal2fov(fy, height)
        fov_x = focal2fov(fx, width)
        viewpoint_camera = GSCamera(
            colmap_id=id, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=camera_transform['img_name'], uid=id,
            image_height=height, image_width=width,)
        return viewpoint_camera
    
    def get_camera(self, path):
        transformsfile ="cameras.json"
        camera_list = []
        circle_cam_list = []
        with open(os.path.join(path, transformsfile)) as json_file:

            camera_transforms = json.load(json_file)
            for idx, frame in enumerate(camera_transforms):
                # if(idx!=0):
                #     continue
                camera_transform = camera_transforms[idx]
                # Extrinsics
                rot = np.array(camera_transform['rotation'])
                pos = np.array(camera_transform['position'])
                W2C = np.zeros((4,4))
                W2C[:3, :3] = rot
                W2C[:3, 3] = pos
                W2C[3,3] = 1
                Rt = np.linalg.inv(W2C)
                T = Rt[:3, 3]
                R = Rt[:3, :3].transpose()
                # Intrinsics
                width = camera_transform['width']
                height = camera_transform['height']
                fy = camera_transform['fy']
                fx = camera_transform['fx']
                fov_y = focal2fov(fy, height)
                fov_x = focal2fov(fx, width)

                viewpoint_camera = GSCamera(
            colmap_id=idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=camera_transform['img_name'], uid=idx,
            image_height=height, image_width=width,)
                if(idx==0):
                    view = viewpoint_camera
                camera_list.append(viewpoint_camera)
        return camera_list
    
    def create_circle_cam(self,path,frames=300):
        transformsfile ="cameras.json"
        camera_list = []
        circle_cam_list = []
        with open(os.path.join(path, transformsfile)) as json_file:
            camera_transforms = json.load(json_file)
            for idx, frame in enumerate(camera_transforms):
                # if(idx!=0):
                #     continue
                camera_transform = camera_transforms[idx]
                # Extrinsics
                rot = np.array(camera_transform['rotation'])
                pos = np.array(camera_transform['position'])
                W2C = np.zeros((4,4))
                W2C[:3, :3] = rot
                W2C[:3, 3] = pos
                W2C[3,3] = 1
                Rt = np.linalg.inv(W2C)
                T = Rt[:3, 3]
                R = Rt[:3, :3].transpose()
                # Intrinsics
                width = camera_transform['width']
                height = camera_transform['height']
                fy = camera_transform['fy']
                fx = camera_transform['fx']
                fov_y = focal2fov(fy, height)
                fov_x = focal2fov(fx, width)
                viewpoint_camera = GSCamera(
            colmap_id=idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=camera_transform['img_name'], uid=idx,
            image_height=height, image_width=width,)
                if(idx==0):
                    view = viewpoint_camera
                camera_list.append(viewpoint_camera)
        circle_pose_list = generate_ellipse_path(camera_list,n_frames=frames)
        
        for idx, pose in enumerate(circle_pose_list):
            view.world_view_transform = jt.array(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view.full_proj_transform = jt.nn.bmm(view.world_view_transform.unsqueeze(0),view.projection_matrix.unsqueeze(0)).squeeze(0)
            view.camera_center = jt.linalg.inv(view.world_view_transform)[3, :3]
            circle_cam_list.append(copy.copy(view))
            # print(view.world_view_transform)
        return circle_cam_list