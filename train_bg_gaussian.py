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
import os
import jittor as jt
from random import randint
from utils.loss_utils import l1_loss, ssim, mesh_restrict_loss
from gaussian_renderer import render, bg_render
import sys
from scene import Scene, GaussianModel, MeshBasedGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from pytorch3d.ops import knn_points,knn_gather

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

jt.flags.use_cuda = 1

# def get_knn(gaussian_points,mesh_points,k=1):
#     #arr: N,3
#     #feature: N,d
#     gaussian_points = gaussian_points.unsqueeze(0).float() #1,N,1
#     mesh_points = mesh_points.unsqueeze(0).float() #1,v,1
#     knn = knn_points(p1=gaussian_points,p2=mesh_points,K=k)
#     return knn

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,mesh_gaussian_model_path,is_exist_bg,remove_neighbor_gaussian_iterations):

    if not is_exist_bg:
        assert False, "Training data must have background!"
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = MeshBasedGaussianModel(dataset.sh_degree)
    bg_gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(mesh_gaussian_model_path)
    scene = Scene(dataset, gaussians,is_exist_bg)
    scene.init_bg_gaussian(bg_gaussians)
    
    bg_gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = jt.load(checkpoint)
        bg_gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = jt.array(bg_color, dtype=jt.float32)

    # iter_start = torch.cuda.Event(enable_timing = True)
    # iter_end = torch.cuda.Event(enable_timing = True)
    if opt.random_background:
        print("use random background to train object, usually for object with background")
    else:
        print("use fixed background to train object, usually for blender object")
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    for iteration in range(first_iter, opt.iterations + 1):
        
        bg_gaussians.update_learning_rate(iteration)
        if iteration < opt.densify_until_iter:
            bg_gaussians.reset_viewspace_point()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            bg_gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = jt.rand((3)) if opt.random_background else background

        render_pkg = bg_render(viewpoint_cam, bg_gaussians, pipe, bg, mesh_gaussians = gaussians)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image


        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        bg_gaussians.optimizer.backward(loss)
        
        if iteration < opt.densify_until_iter:
            viewspace_point_tensor_grad = bg_gaussians.get_viewspace_point_grad()
        update_flag = False

        with jt.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene, bg_render, (pipe, bg), bg_gaussians)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save_bg(iteration)

            # Densification
            if (iteration in remove_neighbor_gaussian_iterations):
                    min_distant = 0.01
                    bg_points = bg_gaussians.get_xyz.clone()
                    mesh_points = gaussians.get_load_xyz.clone()
                    dists,idx = jt.misc.knn(bg_points.unsqueeze(0),mesh_points.unsqueeze(0),k=1)
                    idx = idx.squeeze(0) #N,1
                    dists = dists.squeeze(0) #N,1
                    mask = (dists < min_distant).squeeze()
                    bg_gaussians.prune_points(mask)
                    print("prune success!")
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussian_num = bg_gaussians.get_xyz.shape[0]
                def max(a,b):
                    return jt.where(a>b,a,b)
                bg_gaussians.max_radii2D[visibility_filter[0:gaussian_num]] = max(bg_gaussians.max_radii2D[visibility_filter[0:gaussian_num]], (radii[0:gaussian_num])[visibility_filter[0:gaussian_num]])
                bg_gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                opt.densification_interval = 500
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    size_threshold = None
                    bg_gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    update_flag = True
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    bg_gaussians.reset_opacity()
                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    jt.save((bg_gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        # Optimizer step
        if iteration < opt.iterations:
            # if iteration >= 600:
            #     points_old = gaussians.optimizer.param_groups[0]['params'][0].clone()
            if not update_flag:
                bg_gaussians.optimizer.step()
            # if iteration >= 600:

            #     points_new = gaussians.optimizer.param_groups[0]['params'][0].clone()
            #     print((points_new - points_old).nonzero().shape)
            #     breakpoint()
                
            bg_gaussians.optimizer.zero_grad()

                

            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, bg_gaussians):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = jt.clamp(renderFunc(viewpoint, bg_gaussians, *renderArgs, mesh_gaussians=scene.gaussians)["render"], 0.0, 1.0)
                    gt_image = jt.clamp(viewpoint.original_image, 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None].numpy(), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None].numpy(), global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double().item()
                    psnr_test += psnr(image, gt_image).mean().double().item()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    # breakpoint()
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.bg_gaussians.get_opacity.numpy(), iteration)
            tb_writer.add_scalar('total_points', scene.bg_gaussians.get_xyz.shape[0], iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mesh_gaussian_path", type=str, default ="no mesh")
    parser.add_argument("--is_exist_bg",action='store_true', default=False)
    parser.add_argument("--remove_neighbor_gaussian_iterations", nargs="+", type=int, default=[1_000, 10_000])

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
    args.mesh_gaussian_path, args.is_exist_bg, args.remove_neighbor_gaussian_iterations)
    

    # All done
    print("\nTraining complete.")
