import numpy as np
import jittor as jt
import os
from os import makedirs
import time
import tqdm
import igl
import copy
import numpy as np
import math
import json
from argparse import ArgumentParser, Namespace
from edittool import ObjectVisualTool, SceneVisualTool
from render_origin import save_image
if __name__ == "__main__":
    #the following command only support 1 object deform,a demo
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--is_exist_bg",action='store_true', default=False)
    parser.add_argument("--camera_path",type=str)
    parser.add_argument("--background_gaussian",type=str, default=None)
    parser.add_argument("--object_name",type=str,default="Object")
    parser.add_argument("--object_gaussian",type=str)
    parser.add_argument("--object_origin_mesh",type=str)
    parser.add_argument("--object_deform_mesh",type=str)
    parser.add_argument("--render_path",type=str)
    args = parser.parse_args()
    with jt.no_grad():
        if(args.is_exist_bg):
            scene = SceneVisualTool(args.background_gaussian)
        else:
            scene = ObjectVisualTool()
            
        
        # cams = scene.create_circle_cam(args.camera_path,200) #for video
        cams = scene.get_camera(args.camera_path)
        
        scene.add_gaussian(args.object_gaussian,args.object_origin_mesh,args.object_name)
        scene.deform_one_gaussian(args.object_name, args.object_deform_mesh)
        for i in range(len(cams)):
            cam = cams[i]
            img = scene.render_gaussian(cam)
            os.makedirs(args.render_path, exist_ok=True)
            save_image(img, os.path.join(args.render_path, '{0:05d}'.format(i) + ".png"))
            jt.gc()

### You can make an animation by our code,for example:
        # mesh_sequnce:mesh_path_list
        
        # cam = cams[0]
        # for i in range(len(mesh_sequnce)):
        #     scene.deform_one_gaussian(args.object_name, mesh_sequnce[i])
        #     img = scene.render_gaussian(cam)
        #     os.makedirs(args.render_path, exist_ok=True)
        #     save_image(img, os.path.join(args.render_path, '{0:05d}'.format(i) + ".png"))
        #     jt.gc()





