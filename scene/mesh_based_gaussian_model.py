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
from utils.general_utils import strip_symmetric, build_scaling_rotation, split_mesh_and_gaussian, split_mesh_and_gaussian_pro
import igl

class MeshBasedGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.bc_activation = jt.nn.softmax
        self.scaling_activation = jt.exp
        self.scaling_inverse_activation = jt.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = jt.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.distance_activation = jt.sigmoid
        self.rotation_activation = jt.normalize


    def __init__(self, sh_degree : int,mesh_path = "no mesh"):
        print("Init mesh based gaussian!")
        self.alpha_distance = 4
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._bc = jt.empty(0) #N,3
        self._distance = jt.empty(0) # N,1
        self._features_dc = jt.empty(0)
        self._features_rest = jt.empty(0)
        self._scaling = jt.empty(0)
        self._rotation = jt.empty(0)
        self._opacity = jt.empty(0)
        self.vertex_index = jt.empty(0) # N,3

        self.vertex1 = jt.empty(0) #N,3 coordinate
        self.vertex2 = jt.empty(0)
        self.vertex3 = jt.empty(0)
        self.fid = jt.empty(0) #N,1 index of origin mesh(not split)
        self.normal = jt.empty(0)  # N,3 
        self.r = jt.empty(0) #N,1 restrict gaussian not too far offset the face

        self.max_radii2D = jt.empty(0)
        self.bc_gradient_accum = jt.empty(0)
        self.denom = jt.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.mesh_path = mesh_path

    def capture(self):
        return (
            self.active_sh_degree,
            self._bc,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.bc_gradient_accum,
            self.denom,
            self.vertex1,
            self.vertex2,
            self.vertex3,
            self.normal,
            self._distance,
            self.fid,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._bc, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        bc_gradient_accum, 
        denom,
        self.vertex1,
        self.vertex2,
        self.vertex3,
        self.normal,
        self._distance,
        self.fid,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.bc_gradient_accum = bc_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_bc(self):
        return self._bc

    @property
    def get_number(self):
        return self._bc.shape[0]

    @property
    def get_xyz(self):
        bc = self.bc_activation(self._bc,dim=1)
        n = bc.shape[0]
        # xyz = (bc[:,0].unsqueeze(1))*self.vertex1 + (bc[:,1].unsqueeze(1))*self.vertex2 + (bc[:,2].unsqueeze(1))*self.vertex3
        proj_xyz = jt.unsqueeze(bc[:,0],1)*self.vertex1 + jt.unsqueeze(bc[:,1],1)*self.vertex2 + jt.unsqueeze(bc[:,2],1)*self.vertex3
        # a = jt.unsqueeze(jt.norm((self.vertex1 - self.vertex2),dim=1),1)
        # b = jt.unsqueeze(jt.norm((self.vertex2 - self.vertex3),dim=1),1)
        # c = jt.unsqueeze(jt.norm((self.vertex3 - self.vertex1),dim=1),1)
        # r = (a+b+c)/3

        offset = self.alpha_distance*self.r*(self.distance_activation(self._distance)-0.5)*self.normal
        xyz = proj_xyz + offset

        return xyz
    
    @property
    def get_proj_xyz(self):
        bc = self.bc_activation(self._bc,dim=1)
        n = bc.shape[0]
        # xyz = (bc[:,0].unsqueeze(1))*self.vertex1 + (bc[:,1].unsqueeze(1))*self.vertex2 + (bc[:,2].unsqueeze(1))*self.vertex3
        proj_xyz = jt.unsqueeze(bc[:,0],1)*self.vertex1 + jt.unsqueeze(bc[:,1],1)*self.vertex2 + jt.unsqueeze(bc[:,2],1)*self.vertex3
        return proj_xyz
    
    @property
    def get_load_xyz(self):
        return self.load_xyz

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
        try:
            vertex, triangles = igl.read_triangle_mesh(self.mesh_path)
        except:
            print("Don't have mesh,or don't need mesh if you train the background")
            return 
        num_pts = triangles.shape[0]
        face_normals = igl.per_face_normals(vertex, triangles, np.array([1.0, 0.0, 0.0]))



        fused_point_cloud = (jt.ones((num_pts,3), dtype = jt.float))/3
        distance = (jt.zeros((num_pts,1), dtype = jt.float))
        fused_color = RGB2SH(jt.array(np.random.random((num_pts, 3)), dtype = jt.float))
        features = jt.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype = jt.float)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.vertex1 = jt.array(vertex[triangles[:,0]], dtype = jt.float).stop_grad()
        self.vertex2 = jt.array(vertex[triangles[:,1]], dtype = jt.float).stop_grad()
        self.vertex3 = jt.array(vertex[triangles[:,2]], dtype = jt.float).stop_grad()
        a = jt.unsqueeze(jt.norm((self.vertex1 - self.vertex2),dim=1),1)
        b = jt.unsqueeze(jt.norm((self.vertex2 - self.vertex3),dim=1),1)
        c = jt.unsqueeze(jt.norm((self.vertex3 - self.vertex1),dim=1),1)
        print(a.max())
        print(b.max())
        print(c.max())
        
        self.r = (a+b+c)/3
        print(self.r.max())
        self.fid = jt.unsqueeze(jt.arange(num_pts),1).stop_grad()
        
        self.normal = jt.array(face_normals, dtype = jt.float).stop_grad()
        self.vertex_index = jt.array(triangles).stop_grad()
        self.v = jt.array(vertex).stop_grad()


        p_tmp = (self.vertex1+self.vertex2+self.vertex3)/3
        dist2 = jt.clamp(distCUDA2(p_tmp), min_v=0.0000001)
        scales = jt.log(jt.sqrt(dist2))[...,None].repeat(1, 3)
        rots = jt.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * jt.ones((fused_point_cloud.shape[0], 1), dtype=jt.float32))

        ## jittor init
        self._bc = fused_point_cloud.clone()
        self._distance = distance.clone()
        self._features_dc = features[:,:,0:1].transpose(1, 2).clone()
        self._features_rest = features[:,:,1:].transpose(1, 2).clone()
        self._scaling = scales.clone()
        self._rotation = rots.clone()
        self._opacity = opacities.clone()
        
        self.max_radii2D = jt.zeros((self.get_xyz.shape[0]))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.bc_gradient_accum = jt.zeros((self.get_bc.shape[0], 1))
        self.denom = jt.zeros((self.get_xyz.shape[0], 1))

        l = [
            {'params': [self._bc], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "bc"},
            {'params': [self._distance], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "distance"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = jt.nn.Adam(l, lr=0.0, eps=1e-15)
        self.bc_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def reset_viewspace_point(self):
        self.screenspace_points = jt.zeros_like(self.get_bc, dtype=self.get_xyz.dtype) + 0
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
            if param_group["name"] == "bc":
                lr = self.bc_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "distance":
                param_group['lr'] = lr
        return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz','ca', 'cb', 'cc','v1x','v1y','v1z','v2x','v2y','v2z','v3x','v3y','v3z','dis','v_index1','v_index2','v_index3','radius','face_id']
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

        xyz = self.get_xyz.detach().numpy()
        # normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).numpy()
        opacities = self._opacity.detach().numpy()
        scale = self._scaling.detach().numpy()
        rotation = self._rotation.detach().numpy()
        bc = self._bc.detach().numpy()
        v1 = self.vertex1.detach().numpy()
        v2 = self.vertex2.detach().numpy()
        v3 = self.vertex3.detach().numpy()
        radius = self.r.detach().numpy()
        fid = self.fid.detach().numpy()
        distance = self._distance.detach().numpy()
        normals = self.normal.detach().numpy()

        v_index = self.vertex_index.detach().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, bc, v1, v2, v3, distance, v_index, radius, fid, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        self.load_xyz = self.get_xyz.detach().numpy()

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
        bc = np.stack((np.asarray(plydata.elements[0]["ca"]),
                        np.asarray(plydata.elements[0]["cb"]),
                        np.asarray(plydata.elements[0]["cc"])),  axis=1)
        v1 = np.stack((np.asarray(plydata.elements[0]["v1x"]),
                        np.asarray(plydata.elements[0]["v1y"]),
                        np.asarray(plydata.elements[0]["v1z"])),  axis=1)
        v2 = np.stack((np.asarray(plydata.elements[0]["v2x"]),
                        np.asarray(plydata.elements[0]["v2y"]),
                        np.asarray(plydata.elements[0]["v2z"])),  axis=1)
        v3 = np.stack((np.asarray(plydata.elements[0]["v3x"]),
                        np.asarray(plydata.elements[0]["v3y"]),
                        np.asarray(plydata.elements[0]["v3z"])),  axis=1)
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)
        fid = np.asarray(plydata.elements[0]["face_id"])[..., np.newaxis]
        distance = np.asarray(plydata.elements[0]["dis"])[..., np.newaxis]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        radius = np.asarray(plydata.elements[0]["radius"])[..., np.newaxis]
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

        self.load_xyz = jt.array(xyz, dtype=jt.float)
        self._bc = jt.array(xyz, dtype=jt.float)
        self._features_dc = jt.array(features_dc, dtype=jt.float).transpose(1, 2)
        self._features_rest = jt.array(features_extra, dtype=jt.float).transpose(1, 2)
        self._opacity = jt.array(opacities, dtype=jt.float)
        self._scaling = jt.array(scales, dtype=jt.float)
        self._rotation = jt.array(rots, dtype=jt.float)
        self._distance = jt.array(distance, dtype=jt.float)

        self.vertex1 = jt.array(v1, dtype=jt.float).stop_grad()
        self.vertex2 = jt.array(v2, dtype=jt.float).stop_grad()
        self.vertex3 = jt.array(v3, dtype=jt.float).stop_grad()
        self.normal = jt.array(normal, dtype=jt.float).stop_grad()
        self.r = jt.array(radius, dtype=jt.float).stop_grad()
        self.fid = jt.array(fid, dtype=jt.int).stop_grad()

        self.screenspace_points = jt.zeros_like(self.get_bc, dtype=self.get_bc.dtype) + 0
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

        self._bc = optimizable_tensors["bc"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._distance = optimizable_tensors["distance"]

        self.bc_gradient_accum = self.bc_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.vertex1 = self.vertex1[valid_points_mask]
        self.vertex2 = self.vertex2[valid_points_mask]
        self.vertex3 = self.vertex3[valid_points_mask]
        self.r = self.r[valid_points_mask]
        self.fid = self.fid[valid_points_mask]
        self.normal = self.normal[valid_points_mask]
        self.vertex_index = self.vertex_index[valid_points_mask]

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

    def densification_postfix(self, new_bc, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_distance):
        d = {"bc": new_bc,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "distance" : new_distance}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._bc = optimizable_tensors["bc"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._distance = optimizable_tensors["distance"]

        self.bc_gradient_accum = jt.zeros((self.get_bc.shape[0], 1))
        self.denom = jt.zeros((self.get_bc.shape[0], 1))
        self.max_radii2D = jt.zeros((self.get_bc.shape[0]))

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=4):
        n_init_points = self.get_xyz.shape[0]
        # print("Number of points : ", n_init_points)
        # Extract points that satisfy the gradient condition
        padded_grad = jt.zeros((n_init_points))
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = jt.where(padded_grad >= grad_threshold, True, False)
        # selected_pts_mask = jt.logical_and(selected_pts_mask,
        #                                       jt.max(self.get_scaling, dim=1) > self.percent_dense*scene_extent)
        
        if(selected_pts_mask.sum().item()==0):
            return

        bc = self._bc[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(-1,3)   
        new_bc = jt.ones_like(bc)/3
        distance = self._distance[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(-1,1)
        new_distance = jt.zeros_like(distance) 
        gaussian_num = new_bc.shape[0]
        split_num = self._bc[selected_pts_mask].shape[0]
        #new index
        new_v_index = self.vertex_index[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_v = jt.zeros(split_num,3).unsqueeze(1).repeat(1,3,1)
        new_vertex1 = self.vertex1[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_vertex2 = self.vertex2[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_vertex3 = self.vertex3[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_r = self.r[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)
        new_fid = self.fid[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)
        if(N==4):
            new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index = \
            split_mesh_and_gaussian(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,self.v.shape[0])
        elif(N==5):
            new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index = \
            split_mesh_and_gaussian_pro(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,self.v.shape[0])
        new_vertex1 = new_vertex1.view(gaussian_num, 3)
        new_vertex2 = new_vertex2.view(gaussian_num, 3)
        new_vertex3 = new_vertex3.view(gaussian_num, 3)
        new_v = new_v.view(-1, 3)
        new_v_index = new_v_index.view(gaussian_num, 3)
            
        new_normal = self.normal[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num, 3)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,3) / (4*0.8))
        new_rotation = self._rotation[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,4)
            
        new_features_dc = self._features_dc[selected_pts_mask].unsqueeze(1).repeat(1,N,1,1).view(gaussian_num,-1,3)
        new_features_rest = self._features_rest[selected_pts_mask].unsqueeze(1).repeat(1,N,1,1).view(gaussian_num,-1,3)
        new_opacity = self._opacity[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)

        self.densification_postfix(new_bc, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_distance)

        self.vertex1 = jt.concat((self.vertex1, new_vertex1), dim=0)
        self.vertex2 = jt.concat((self.vertex2, new_vertex2), dim=0)
        self.vertex3 = jt.concat((self.vertex3, new_vertex3), dim=0)
        self.vertex_index = jt.concat((self.vertex_index, new_v_index), dim=0)
        self.r = jt.concat((self.r, new_r), dim=0)
        self.v = jt.concat((self.v, new_v), dim=0)
        self.normal = jt.concat((self.normal, new_normal),dim=0)
        self.fid = jt.concat((self.fid, new_fid),dim=0)
        
        prune_filter = jt.concat((selected_pts_mask, jt.zeros(N * selected_pts_mask.sum().item(), dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        return
        # Extract points that satisfy the gradient condition
        # selected_pts_mask = jt.where(jt.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask = jt.logical_and(selected_pts_mask,
        #                                       jt.max(self.get_scaling, dim=1) <= self.percent_dense*scene_extent)
        
        # new_xyz = self._xyz[selected_pts_mask]
        # new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        # new_opacities = self._opacity[selected_pts_mask]
        # new_scaling = self._scaling[selected_pts_mask]
        # new_rotation = self._rotation[selected_pts_mask]

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, densification_postfix)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, N = 4):
        grads = self.bc_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_split(grads, max_grad, extent, N)
        jt.gc()

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.bc_gradient_accum[update_filter] += jt.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def save_mesh(self,path):
        vertices = self.v.numpy()
        faces = self.vertex_index.numpy()
        igl.write_triangle_mesh(path, vertices, faces)

    def densify_and_split_for_init(self, N=4):
        n_init_points = self.get_bc.shape[0]
        print("after init the face number is:", n_init_points*N)

        # split all points
        selected_pts_mask = jt.ones((n_init_points),dtype=jt.bool)

        bc = self._bc[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(-1,3)   
        new_bc = jt.ones_like(bc)/3
        distance = self._distance[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(-1,1)
        new_distance = jt.zeros_like(distance) 
        gaussian_num = new_bc.shape[0]
        split_num = self._bc[selected_pts_mask].shape[0]
        #new index
        new_v_index = self.vertex_index[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_v = jt.zeros(split_num,3).unsqueeze(1).repeat(1,3,1)
        new_vertex1 = self.vertex1[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_vertex2 = self.vertex2[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_vertex3 = self.vertex3[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_fid = self.fid[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)
        new_r = self.r[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)
        new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index = \
            split_mesh_and_gaussian(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,self.v.shape[0])
       
        new_vertex1 = new_vertex1.view(gaussian_num, 3)
        new_vertex2 = new_vertex2.view(gaussian_num, 3)
        new_vertex3 = new_vertex3.view(gaussian_num, 3)
        new_v = new_v.view(-1, 3)
        new_v_index = new_v_index.view(gaussian_num, 3)
            
        new_normal = self.normal[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num, 3)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,3) / (4*0.8))
        new_rotation = self._rotation[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,4)
            
        new_features_dc = self._features_dc[selected_pts_mask].unsqueeze(1).repeat(1,N,1,1).view(gaussian_num,-1,3)
        new_features_rest = self._features_rest[selected_pts_mask].unsqueeze(1).repeat(1,N,1,1).view(gaussian_num,-1,3)
        new_opacity = self._opacity[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)

        self.densification_postfix(new_bc, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_distance)

        self.vertex1 = jt.concat((self.vertex1, new_vertex1), dim=0)
        self.vertex2 = jt.concat((self.vertex2, new_vertex2), dim=0)
        self.vertex3 = jt.concat((self.vertex3, new_vertex3), dim=0)
        self.vertex_index = jt.concat((self.vertex_index, new_v_index), dim=0)
        self.r = jt.concat((self.r, new_r), dim=0)
        self.v = jt.concat((self.v, new_v), dim=0)
        self.normal = jt.concat((self.normal, new_normal),dim=0)
        self.fid = jt.concat((self.fid, new_fid),dim=0)

        prune_filter = jt.concat((selected_pts_mask, jt.zeros(N * n_init_points, dtype=bool)))
        self.prune_points(prune_filter)
        jt.gc()