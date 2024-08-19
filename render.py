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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, bg_render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import MeshBasedGaussianModel, GaussianModel
import cv2
import numpy as np

jt.flags.use_cuda = 1


def save_image(mat,path):
    mat = mat.transpose(1,2,0)
    mat = mat[:,:,[2,1,0]].clamp(0,1) * 255
    cv2.imwrite(path,mat.numpy().astype(np.uint8))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, bg_gaussian):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if(bg_gaussian==None):
            rendering = render(view, gaussians, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]*view.mask + (jt.unsqueeze(jt.unsqueeze(background,1),1)) * (1-view.mask)
        else:
            rendering = bg_render(view, bg_gaussian, pipeline, background, mesh_gaussians=gaussians)["render"]
            gt = view.original_image[0:3, :, :]*view.mask + (jt.unsqueeze(jt.unsqueeze(background,1),1)) * (1-view.mask)
        save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, is_exist_bg : bool, bg_gaussian_path : str):
    with jt.no_grad():
        bg_gaussian = None
        if(bg_gaussian_path!="no bg"):
            bg_gaussian = GaussianModel(dataset.sh_degree)
            bg_gaussian.load_ply(bg_gaussian_path)
        gaussians = MeshBasedGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, is_exist_bg, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = jt.array(bg_color, dtype=jt.float32)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, bg_gaussian)

        if not skip_test:
            
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, bg_gaussian)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--bg_gaussian_path", type=str, default = "no bg")
    parser.add_argument("--is_exist_bg",action='store_true', default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.is_exist_bg, args.bg_gaussian_path)