
#!/bin/bash
python train_mesh_gaussian.py -r 1 -s /mnt/sda/zhangbotao/blendedmvs/excavator -m /mnt/sda/zhangbotao/jittor-output/excavator  --is_exist_bg --input_mesh /mnt/sda/zhangbotao/blendedmvs/excavator/deform/1.obj
python train_bg_gaussian.py -r 1 -s /mnt/sda/zhangbotao/blendedmvs/excavator -m /mnt/sda/zhangbotao/jittor-output/excavator --mesh_gaussian_path /mnt/sda/zhangbotao/jittor-output/excavator/point_cloud/iteration_30000/point_cloud.ply  --is_exist_bg
python render_origin.py -m /mnt/sda/zhangbotao/jittor-output/excavator -s /mnt/sda/zhangbotao/blendedmvs/excavator
python edit.py --camera_path /mnt/sda/zhangbotao/jittor-output/excavator --object_name excavator --object_gaussian /mnt/sda/zhangbotao/jittor-output/excavator/point_cloud/iteration_30000/point_cloud.ply \
--object_origin_mesh /mnt/sda/zhangbotao/blendedmvs/excavator/deform/1.obj \
--object_deform_mesh /mnt/sda/zhangbotao/blendedmvs/excavator/deform/4.obj \
--render_path /mnt/sda/zhangbotao/jittor-output/excavator/deform \
--is_exist_bg --background_gaussian /mnt/sda/zhangbotao/jittor-output/excavator/point_cloud/iteration_30000/bg_point_cloud.ply

