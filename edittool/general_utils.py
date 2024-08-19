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


def inverse_sigmoid(x):
    return jt.log(x/(1-x))

def PILtoJittor(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = jt.array(np.array(resized_image_PIL),dtype='float32') / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

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

def get_barycentric_coordinate(gaussians,p1,p2,p3):
    e1 = gaussians-p1
    e2 = gaussians-p2
    e3 = gaussians-p3
    s1 = np.cross(e2,e3)
    s1 = np.linalg.norm(s1,axis=1)
    s2 = np.cross(e1,e3)
    s2 = np.linalg.norm(s2,axis=1)
    s3 = np.cross(e1,e2)
    s3 = np.linalg.norm(s3,axis=1)
    s1 = np.expand_dims(s1,axis=1)
    s2 = np.expand_dims(s2,axis=1)
    s3 = np.expand_dims(s3,axis=1)
    s = s1+s2+s3
    coord = np.concatenate([s1/s,s2/s,s3/s],axis =1)
    return coord