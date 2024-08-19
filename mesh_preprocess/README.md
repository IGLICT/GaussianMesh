**if you use instant-nsr-pl to train colmap dataset with mask，following the step**:

1.Copy colmap.py to instant-nsr-pl/dataset/colmap.py

- You can get **transform_matrix** and **scaling_factor** when running code(Line94,95 in colmap.py).
- Remember or save the **transform_matrix** and **scaling_factor**(npy format is OK)

2.Turn to /instant-nsr-pl/configs/neus-colmap.yaml,

- Change   “center_est_method: point"
- Change  "apply_mask: true"

3.Running code reference official readme (You can see the **transform_matrix** and **scaling_factor**)

4.Running script "convert_mesh.py",get the reconstructed mesh in original space. 

5.Running MeshFix to fixmesh,put the mesh.obj to  (<data_name>/mesh_sequnce).Or other position is OK.

