import numpy as np
import igl

def apply_transformation(mesh_path, transformation_matrix, scale, save_path):
    # Load the mesh using igl
    v, f = igl.read_triangle_mesh(mesh_path)
    v = v*scale
    # Append a column of ones to the vertices to make them homogeneous coordinates
    ones_column = np.ones((v.shape[0], 1))
    v_homogeneous = np.hstack((v, ones_column))
    # Apply the transformation matrix to each vertex
    transformed_vertices = np.dot(v_homogeneous, transformation_matrix.T)
    # Extract the transformed vertices
    transformed_vertices = transformed_vertices[:, :3]
    # Create a new mesh with the transformed vertices
    # Save the transformed mesh
    igl.write_triangle_mesh(save_path, transformed_vertices,f)
    print("save mesh to {} success".format(save_path))

if __name__ == "__main__":
    # Example transformation matrix (you should replace this with your actual 4x4 matrix)
    transformation_matrix = np.array([[ 5.0978e-03,  1.6974e-01,  9.8548e-01, -4.4309e-01],
        [-9.9955e-01,  3.0019e-02,  4.2092e-10, -5.8854e-02],
        [-2.9583e-02, -9.8503e-01,  1.6982e-01, -3.7272e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    transformation_matrix = np.linalg.inv(transformation_matrix)
    # Example scale factor (you should replace this with your actual 4x4 matrix)
    scale = 1.9550

    # Replace 'your_mesh.obj' with the path to your mesh file
    mesh_path = r"\horse\it-20000.obj"
    save_path = r"\horse\aligned_mesh.obj"
    apply_transformation(mesh_path, transformation_matrix,scale,save_path)