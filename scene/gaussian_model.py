#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import jittor as jt

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from jittor import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from scene.simple_knn import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = jt.exp
        self.scaling_inverse_activation = jt.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = jt.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = jt.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = jt.empty(0)
        self._features_dc = jt.empty(0)
        self._features_rest = jt.empty(0)
        self._scaling = jt.empty(0)
        self._rotation = jt.empty(0)
        self._opacity = jt.empty(0)
        self.max_radii2D = jt.empty(0)
        self.xyz_gradient_accum = jt.empty(0)
        self.denom = jt.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return jt.concat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        
        self.spatial_lr_scale = spatial_lr_scale
        print("start")

        fused_point_cloud = jt.array(np.asarray(pcd.points)).float()
        
        fused_color = RGB2SH(jt.array(np.asarray(pcd.colors)).float())
        
        features = jt.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # np.save("input_pcd.npy",np.asarray(pcd.points))
        dist2 = jt.clamp(distCUDA2(jt.array(np.asarray(pcd.points)).float()), min_v=0.0000001)
        scales = jt.log(jt.sqrt(dist2))[...,None].repeat(1, 3)
        rots = jt.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * jt.ones((fused_point_cloud.shape[0], 1), dtype=jt.float32))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        ## jittor init
        self._xyz = fused_point_cloud.clone()
        self._features_dc = features[:,:,0:1].transpose(1, 2).clone()
        self._features_rest = features[:,:,1:].transpose(1, 2).clone()
        self._scaling = scales.clone()
        self._rotation = rots.clone()
        self._opacity = opacities.clone()
        
        self.max_radii2D = jt.zeros((self.get_xyz.shape[0]))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = jt.zeros((self.get_xyz.shape[0], 1))
        self.denom = jt.zeros((self.get_xyz.shape[0], 1))

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = jt.nn.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def reset_viewspace_point(self):
        self.screenspace_points = jt.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype) + 0
        pg = self.optimizer.param_groups[-1]
        if pg["name"] == "screenspace_points":
            self.optimizer.param_groups.pop()
        self.optimizer.add_param_group(
            {'params': [self.screenspace_points], 'lr':0., "name": "screenspace_points"}
        )
    def get_viewspace_point_grad(self):
        pg = self.optimizer.param_groups[-1]
        if pg["name"] == "screenspace_points":
            # breakpoint()
            return pg["grads"][0]
        else:
            assert False, "No viewspace_point_grad found"
            
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).numpy()
        opacities = self._opacity.detach().numpy()
        scale = self._scaling.detach().numpy()
        rotation = self._rotation.detach().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        def min(a,b):
            return jt.where(a<b,a,b)
        opacities_new = inverse_sigmoid(min(self.get_opacity, jt.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = jt.array(xyz, dtype=jt.float)
        self._features_dc = jt.array(features_dc, dtype=jt.float).transpose(1, 2)
        self._features_rest = jt.array(features_extra, dtype=jt.float).transpose(1, 2)
        self._opacity = jt.array(opacities, dtype=jt.float)
        self._scaling = jt.array(scales, dtype=jt.float)
        self._rotation = jt.array(rots, dtype=jt.float)
        self.screenspace_points = jt.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype) + 0
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'screenspace_points': 
                continue
            if group["name"] == name:
                with jt.enable_grad():
                    group["params"][0] = tensor.copy()
                group["m"][0] = jt.zeros_like(tensor)
                group["values"][0] = jt.zeros_like(tensor)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'screenspace_points': 
                continue
            if group['params'][0] is not None:
                group['m'][0].update(group['m'][0][mask])
                group['values'][0].update(group['values'][0][mask])
                with jt.enable_grad():
                    old = group["params"].pop()
                    group["params"].append(old[mask])
                    del old
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = mask.logical_not()
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == 'screenspace_points': 
                continue
            extension_tensor = tensors_dict[group["name"]]
            group["m"][0] = jt.concat((group["m"][0], jt.zeros_like(extension_tensor)), dim=0)
            
            group["values"][0] = jt.concat((group["values"][0], jt.zeros_like(extension_tensor)), dim=0)
            old_tensor = group["params"].pop()
            with jt.enable_grad():
                group["params"].append(jt.concat((old_tensor, extension_tensor), dim=0))
                del old_tensor
            optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = jt.zeros((self.get_xyz.shape[0], 1))
        self.denom = jt.zeros((self.get_xyz.shape[0], 1))
        self.max_radii2D = jt.zeros((self.get_xyz.shape[0]))

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # print("Number of points : ", n_init_points)
        # Extract points that satisfy the gradient condition
        padded_grad = jt.zeros((n_init_points))
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = jt.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1) > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = jt.zeros((stds.size(0), 3))
        samples = jt.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = jt.nn.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = jt.concat([selected_pts_mask, jt.zeros((N * selected_pts_mask.sum().item(),), dtype=bool)],dim=0)
        self.prune_points(prune_filter)
        

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = jt.where(jt.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1) <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1) > 0.1 * extent
            prune_mask = jt.logical_or(jt.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        jt.gc()
        

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        num = self.get_xyz.shape[0]
        update_filter = update_filter[0:num]
        self.xyz_gradient_accum[update_filter] += jt.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        