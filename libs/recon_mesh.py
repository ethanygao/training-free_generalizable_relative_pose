import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
import torch

def mesh_from_depth(depth_map, input_rgb, texture_pca, cam_K=None, 
                    transformation=None, max_depth_diff_ratio=4, scale_factor=1.0,
                    cache_rgb_mesh_path = None, cache_pca_mesh_path = None):
    nonzero_indices = np.argwhere(depth_map > 0)
    points_2d = nonzero_indices[:, ::-1].astype(np.float32)
    
    tri = Delaunay(points_2d)

    if cam_K is not None:
        width = depth_map.shape[1]
        height = depth_map.shape[0]
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, cam_K[0][0], cam_K[1][1], cam_K[0][2], cam_K[1][2]
        )
        points_3d = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_map), camera_intrinsics, depth_scale=scale_factor
        )
    else:
        depth_values = depth_map[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        points_3d = np.column_stack([nonzero_indices[:, 1], nonzero_indices[:, 0], depth_values])

    if transformation is not None:
        points_3d = points_3d.transform(transformation)
    
    points_3d = np.asarray(points_3d.points) if hasattr(points_3d, 'points') else points_3d

    tri_vertices = points_3d[tri.simplices]
    z_coords = tri_vertices[:, :, 2]

    diff01 = np.abs(z_coords[:, 0] - z_coords[:, 1])
    diff02 = np.abs(z_coords[:, 0] - z_coords[:, 2])
    diff12 = np.abs(z_coords[:, 1] - z_coords[:, 2])
    
    diffs = np.concatenate([diff01, diff02, diff12])
    median_distance = np.percentile(diffs, 75)
    max_depth_diff = max_depth_diff_ratio * median_distance
    
    mask = (diff01 < max_depth_diff) & (diff02 < max_depth_diff) & (diff12 < max_depth_diff)
    triangles = tri.simplices[mask]
    
    texture_rgb = np.asarray(input_rgb)
    texture_rgb_ = texture_rgb[nonzero_indices[:, 0], nonzero_indices[:, 1]]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_3d)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.filter_smooth_laplacian(number_of_iterations=10, lambda_filter=100)
    
    texture_pca = np.asarray(texture_pca)
    texture_pca_ = texture_pca[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    
    vtx_pos = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    pos_idx = torch.tensor(np.array(mesh.triangles), dtype=torch.int32)
    rgb_vtx_col = torch.tensor(texture_rgb_ / 255.0, dtype=torch.float32)
    pca_vtx_col = torch.tensor(texture_pca_ / 255.0, dtype=torch.float32)
    
    if cache_rgb_mesh_path is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(texture_rgb_/255.0) 
        o3d.io.write_triangle_mesh(cache_rgb_mesh_path, mesh)
    
    if cache_pca_mesh_path is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(texture_rgb_/255.0) 
        o3d.io.write_triangle_mesh(cache_pca_mesh_path, mesh)
        
    return vtx_pos, pos_idx, rgb_vtx_col, pca_vtx_col

