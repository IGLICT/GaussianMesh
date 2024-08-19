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

from jittor import nn
import os
from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation, inverse_sigmoid


class MeshBasedGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            # symm = strip_symmetric(actual_covariance)
            return actual_covariance

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
        proj_xyz = jt.unsqueeze(bc[:,0],1)*self.vertex1 + jt.unsqueeze(bc[:,1],1)*self.vertex2 + jt.unsqueeze(bc[:,2],1)*self.vertex3
        offset =self.alpha_distance*self.r*(self.distance_activation(self._distance)-0.5)*self.normal
        xyz = proj_xyz + offset
        return xyz
    
    @property
    def get_proj_xyz(self):
        bc = self.bc_activation(self._bc,dim=1)
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