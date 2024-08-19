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

from pathlib import Path
import os
from PIL import Image
import jittor as jt

from utils.loss_utils import ssim
from lpips_jittor import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

jt.flags.use_cuda=1

lpips_func = lpips.LPIPS(net='vgg')
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    to_tensor = jt.transform.ToTensor()
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        # breakpoint()
        renders.append(jt.array(to_tensor(render)).unsqueeze(0)[:, :3, :, :])
        gts.append(jt.array(to_tensor(gt)).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"
        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            
            renders, gts, image_names = readImages(renders_dir, gt_dir)
            # print(len(renders))

            ssims = []
            psnrs = []
            lpipss = []
            with jt.no_grad():
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_func(renders[idx], gts[idx]).sync())
                

            print("  SSIM : {:>12.7f}".format(jt.array(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(jt.array(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(jt.array(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": jt.array(ssims).mean().item(),
                                                    "PSNR": jt.array(psnrs).mean().item(),
                                                    "LPIPS": jt.array(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(jt.array(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(jt.array(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(jt.array(lpipss).tolist(), image_names)}})

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
