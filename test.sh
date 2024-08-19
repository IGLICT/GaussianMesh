
#!/bin/bash
# python train_mesh_gaussian.py -r 1 -s /mnt/sda/zhangbotao/garden -m output/garden  --is_exist_bg --input_mesh /mnt/sda/zhangbotao/garden/mvs_deform1/1.obj
# python train_bg_gaussian.py -r 1 -s /mnt/sda/zhangbotao/garden -m output/garden --mesh_gaussian_path output/garden/point_cloud/iteration_30000/point_cloud.ply  --is_exist_bg
# python render_origin.py -m output/garden -s /mnt/sda/zhangbotao/garden
# python edit.py --camera_path output/garden --object_name garden-table --object_gaussian output/garden/point_cloud/iteration_30000/point_cloud.ply \
# --object_origin_mesh /mnt/sda/zhangbotao/garden/mvs_deform1/1.obj \
# --object_deform_mesh /mnt/sda/zhangbotao/garden/mvs_deform1/3.obj \
# --render_path output/garden/deform \
# --is_exist_bg --background_gaussian output/garden/point_cloud/iteration_30000/bg_point_cloud.ply

python train_mesh_gaussian.py -r 1 -s /mnt/sda/zhangbotao/case2 -m output/square --input_mesh /mnt/sda/zhangbotao/case2/1.obj
CUDA_VISIBLE_DEVICES=1 python render_origin.py -m output/square -s /mnt/sda/zhangbotao/case2
python edit.py --camera_path output/square --object_name square --object_gaussian output/square/point_cloud/iteration_7000/point_cloud.ply \
--object_origin_mesh /mnt/sda/zhangbotao/case2/1.obj \
--object_deform_mesh /mnt/sda/zhangbotao/case2/4.obj \
--render_path output/square/deform
