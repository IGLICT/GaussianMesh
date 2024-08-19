import os
import json

import numpy as np
import jittor as jt
from jittor import nn
from PIL import Image

from edittool.graphics_utils import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix




class GSCamera(nn.Module):
    """Class to store Gaussian Splatting camera parameters.
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_height=None, image_width=None,
                 ):
        """
        Args:
            colmap_id (int): ID of the camera in the COLMAP reconstruction.
            R (np.array): Rotation matrix.
            T (np.array): Translation vector.
            FoVx (float): Field of view in the x direction.
            FoVy (float): Field of view in the y direction.
            image (np.array): GT image.
            gt_alpha_mask (_type_): _description_
            image_name (_type_): _description_
            uid (_type_): _description_
            trans (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
            scale (float, optional): _description_. Defaults to 1.0.
            data_device (str, optional): _description_. Defaults to "cuda".
            image_height (_type_, optional): _description_. Defaults to None.
            image_width (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super(GSCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name


        if image is None:
            if image_height is None or image_width is None:
                raise ValueError("Either image or image_height and image_width must be specified")
            else:
                self.image_height = image_height
                self.image_width = image_width
        else:        
            self.original_image = image.clamp(0.0, 1.0)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask
            else:
                self.original_image *= jt.ones((1, self.image_height, self.image_width))

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = jt.array(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = jt.nn.bmm(self.world_view_transform.unsqueeze(0),self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = jt.linalg.inv(self.world_view_transform)[3, :3]
