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

import jittor as jt
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return jt.log(x/(1-x))

def PILtoJittor(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = jt.array(np.array(resized_image_PIL),dtype='float32') / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = jt.zeros((L.shape[0], 6), dtype=jt.float32)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = jt.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = jt.zeros((q.size(0), 3, 3), dtype=jt.float32)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = jt.zeros((s.shape[0], 3, 3), dtype=jt.float32)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    jt.set_seed(0)

def split_mesh_and_gaussian(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,v_origin_num):
# in:new_vertex1:[N,4,3],new_v_index[N,4,3],new_v:[N,3,3],v_origin_num:原来顶点的个数
# out:[N,4,3]
    
    a = new_vertex1[:,0,:].clone()
    b = new_vertex2[:,0,:].clone()
    c = new_vertex3[:,0,:].clone()
    new_vertex1[:,0,:] = a
    new_vertex1[:,1,:] = (a+b)/2
    new_vertex1[:,2,:] = (a+c)/2
    new_vertex1[:,3,:] = (a+b)/2
    new_vertex2[:,0,:] = (a+b)/2
    new_vertex2[:,1,:] = b
    new_vertex2[:,2,:] = (c+b)/2
    new_vertex2[:,3,:] = (b+c)/2
    new_vertex3[:,0,:] = (a+c)/2
    new_vertex3[:,1,:] = (c+b)/2
    new_vertex3[:,2,:] = c
    new_vertex3[:,3,:] = (a+c)/2

    new_v[:,0,:] = (a+b)/2
    new_v[:,1,:] = (a+c)/2
    new_v[:,2,:] = (b+c)/2

    tmp = jt.arange(new_v.shape[0]*3).view(new_v.shape[0],3)
    

    new_v_index[:,0,1] = tmp[:,0] + v_origin_num
    new_v_index[:,0,2] = tmp[:,1] + v_origin_num
    new_v_index[:,1,0] = tmp[:,0] + v_origin_num
    new_v_index[:,1,2] = tmp[:,2] + v_origin_num
    new_v_index[:,2,0] = tmp[:,1] + v_origin_num
    new_v_index[:,2,1] = tmp[:,2] + v_origin_num
    new_v_index[:,3,0] = tmp[:,0] + v_origin_num
    new_v_index[:,3,1] = tmp[:,2] + v_origin_num
    new_v_index[:,3,2] = tmp[:,1] + v_origin_num

    return new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index

def split_mesh_and_gaussian_pro(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,v_origin_num):
# in:new_vertex1:[N,5,3],new_v_index[N,5,3],new_v:[N,3,3],v_origin_num:原来顶点的个数
# out:[N,5,3]
    
    a = new_vertex1[:,0,:].clone()
    b = new_vertex2[:,0,:].clone()
    c = new_vertex3[:,0,:].clone()
    new_vertex1[:,0,:] = a
    new_vertex1[:,1,:] = (a+b)/2
    new_vertex1[:,2,:] = (a+c)/2
    new_vertex1[:,3,:] = (a+b)/2
    new_vertex2[:,0,:] = (a+b)/2
    new_vertex2[:,1,:] = b
    new_vertex2[:,2,:] = (c+b)/2
    new_vertex2[:,3,:] = (b+c)/2
    new_vertex3[:,0,:] = (a+c)/2
    new_vertex3[:,1,:] = (c+b)/2
    new_vertex3[:,2,:] = c
    new_vertex3[:,3,:] = (a+c)/2
    new_vertex1[:,4,:] = a
    new_vertex2[:,4,:] = b
    new_vertex3[:,4,:] = c

    new_v[:,0,:] = (a+b)/2
    new_v[:,1,:] = (a+c)/2
    new_v[:,2,:] = (b+c)/2

    tmp = jt.arange(new_v.shape[0]*3).view(new_v.shape[0],3)
    

    new_v_index[:,0,1] = tmp[:,0] + v_origin_num
    new_v_index[:,0,2] = tmp[:,1] + v_origin_num
    new_v_index[:,1,0] = tmp[:,0] + v_origin_num
    new_v_index[:,1,2] = tmp[:,2] + v_origin_num
    new_v_index[:,2,0] = tmp[:,1] + v_origin_num
    new_v_index[:,2,1] = tmp[:,2] + v_origin_num
    new_v_index[:,3,0] = tmp[:,0] + v_origin_num
    new_v_index[:,3,1] = tmp[:,2] + v_origin_num
    new_v_index[:,3,2] = tmp[:,1] + v_origin_num

    return new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index