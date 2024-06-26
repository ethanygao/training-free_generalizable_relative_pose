import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

## reconstruct the object mesh using Delaunay
def mesh_from_depth(depth_map, input_texture, save_path, cam_K = None, transformation = None, max_depth_diff_ratio = 4, scale_factor=1.0):
    '''
    params:
        depth_path: depth_path of the depth image
        rgb_path: rgb_path of the image
        save_path: the path to save the ply file
        cam_K: camera intrinsic matrix in shape of (3,3)
        transformation: the c2m transformation
        mex_depth_diff: the threshold for the depth difference
        sacle_factor: the depth scale factor
        
    return: the reconstructed objcet mesh   
    '''
    nonzero_indices = np.argwhere(depth_map > 0)
    points_2d = nonzero_indices[:, ::-1]

    tri = Delaunay(points_2d)

    if cam_K is not None:
        width = depth_map.shape[1]
        height = depth_map.shape[0]
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, cam_K[0][0], cam_K[1][1], cam_K[0][2], cam_K[1][2])
        points_3d = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_map), camera_intrinsics, depth_scale = scale_factor)
    else:
        depth_values = depth_map[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        points_3d = np.column_stack([nonzero_indices[:, 1], nonzero_indices[:, 0], depth_values])
        
    if transformation is not None:
        points_3d = points_3d.transform(transformation)
    
    points_3d = points_3d.points
        
    distances = []
    for simplex in tri.simplices:
        p0, p1, p2 = simplex
        point0 = points_3d[p0]
        point1 = points_3d[p1]
        point2 = points_3d[p2]

        depth_diff01 = abs(point0[2] - point1[2])
        depth_diff02 = abs(point0[2] - point2[2])
        depth_diff12 = abs(point1[2] - point2[2])
            
        distances.append(depth_diff01)
        distances.append(depth_diff02)
        distances.append(depth_diff12)

    median_distance = np.percentile(distances, 75)
    max_depth_diff = max_depth_diff_ratio * median_distance

    triangles = []
    for simplex in tri.simplices:
        p0, p1, p2 = simplex
        point0 = points_3d[p0]
        point1 = points_3d[p1]
        point2 = points_3d[p2]
        
        depth_diff01 = abs(point0[2] - point1[2])
        depth_diff02 = abs(point0[2] - point2[2])
        depth_diff12 = abs(point1[2] - point2[2])
        
        if depth_diff01 < max_depth_diff and depth_diff02 < max_depth_diff and depth_diff12 < max_depth_diff:
            triangles.append([p0, p1, p2])

    texture = np.asarray(input_texture)
    texture_ = texture[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_3d)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.filter_smooth_laplacian(number_of_iterations=10, lambda_filter=100)
    mesh.vertex_colors = o3d.utility.Vector3dVector(texture_/255.0) 
    mesh.compute_vertex_normals()
    
    o3d.io.write_triangle_mesh(save_path, mesh)
    
    return mesh

    