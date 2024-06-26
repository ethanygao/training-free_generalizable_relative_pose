import numpy as np
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d
import os
import csv
import math

#----------------------------------------------------------------------------
### dataloader utils ###
#----------------------------------------------------------------------------
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_image_crop_resize(image, box, resize_shape):
    """Crop image according to the box, and resize the cropped image to resize_shape
    @param image: the image waiting to be cropped
    @param box: [x0, y0, x1, y1]
    @param resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    scale = np.array([box[2] - box[0], box[3] - box[1]])

    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    image_crop = cv2.warpAffine(
        image, trans_crop, (resize_w, resize_h), flags=cv2.INTER_LINEAR)

    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)
    return image_crop, trans_crop_homo

def get_K_crop_resize(box, K_orig, resize_shape):
    """Update K (crop an image according to the box, and resize the cropped image to resize_shape) 
    @param box: [x0, y0, x1, y1]
    @param K_orig: [3, 3] or [3, 4]
    @resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    scale = np.array([box[2] - box[0], box[3] - box[1]])  # w, h

    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)

    if K_orig.shape == (3, 3):
        K_orig_homo = np.concatenate([K_orig, np.zeros((3, 1))], axis=-1)
    else:
        K_orig_homo = K_orig.copy()
    assert K_orig_homo.shape == (3, 4)

    K_crop_homo = trans_crop_homo @ K_orig_homo  # [3, 4]
    K_crop = K_crop_homo[:3, :3]

    return K_crop, K_crop_homo

#----------------------------------------------------------------------------
### render utils ###
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Quaternion math.
#----------------------------------------------------------------------------
# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)

# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]
def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q) # Compute the dot product between p and q
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d) # Compute the angle between p and q
    tt = t0 * t # Compute the interpolated angle
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    # q_slerp(q1, q2, t) 用于在两个四元数 q1 和 q2 之间进行球面线性插值，参数 t 为插值的权重
    return q_slerp(q_unit(), q, scl)

# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

def q_angle_deg_batch(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()

    # Compute dot product
    dot_product = np.sum(p * q, axis=1)

    # Ensure the dot product is within [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute angular difference in degrees
    angle_degrees = np.degrees(2.0 * np.arccos(dot_product))

    return angle_degrees

# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix.
def q_to_mtx(q):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [400]], dtype=torch.float32).cuda()], dim=1) # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    return rr

#  Returns torch.sqrt(torch.max(0, x))  but with a zero subgradient where x is 0.
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:

    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

#  Returns torch.sqrt(torch.max(0, x))  but with a zero subgradient where x is 0.
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:

    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

# ------------------------------------------------------
# Quatnion Operation, From EproPnP
# ------------------------------------------------------
def skew(x):
    """
    Args:
        x (torch.Tensor): shape (*, 3)

    Returns:
        torch.Tensor: (*, 3, 3), skew symmetric matrices
    """
    mat = x.new_zeros(x.shape[:-1] + (3, 3))
    mat[..., [2, 0, 1], [1, 2, 0]] = x
    mat[..., [1, 2, 0], [2, 0, 1]] = -x
    return mat

def quaternion_to_rot_mat(quaternions):
    """
    Args:
        quaternions (torch.Tensor): (*, 4)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    if quaternions.requires_grad:
        w, i, j, k = torch.unbind(quaternions, -1)
        rot_mats = torch.stack((
            1 - 2 * (j * j + k * k),     2 * (i * j - k * w),     2 * (i * k + j * w),
                2 * (i * j + k * w), 1 - 2 * (i * i + k * k),     2 * (j * k - i * w),
                2 * (i * k - j * w),     2 * (j * k + i * w), 1 - 2 * (i * i + j * j)), dim=-1,
        ).reshape(quaternions.shape[:-1] + (3, 3))
    else:
        w, v = quaternions.split([1, 3], dim=-1)
        rot_mats = 2 * (w.unsqueeze(-1) * skew(v) + v.unsqueeze(-1) * v.unsqueeze(-2))
        diag = torch.diagonal(rot_mats, dim1=-2, dim2=-1)
        diag += w * w - (v.unsqueeze(-2) @ v.unsqueeze(-1)).squeeze(-1)
    return rot_mats

# Convert rotations given as rotation matrices to quaternions.
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
  
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


# load singel mesh information.
def load_ply_mesh(filename, device=torch.device('cpu')):
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.triangles, dtype=torch.int32, device=device)

    if mesh.has_vertex_colors():
        colors = torch.tensor(mesh.vertex_colors, dtype=torch.float32, device=device)
    else:
        colors = None
    
    return {"v": vertices, "f": faces, "colors": colors}

# Rotational Error.
def rotation_error(R_est, R_gt):
    """
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated rotation error in [deg].
    """
    assert (R_est.shape == R_gt.shape == (3, 3))
    
    if torch.is_tensor(R_est):
        R_est = R_est.detach().cpu().numpy()
        R_gt = R_gt.detach().cpu().numpy()
    
    # cos(θ)= 0.5 * (trace(Rest*Rgt^−1)−1)
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))
    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error


#----------------------------------------------------------------------------
### main utils ###
#----------------------------------------------------------------------------
def check_exists(csv_path, qry_ref_name, scene_name):
    with open(os.path.join(csv_path), 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        data.pop(0)
   
    first_column = list(map(str, [row[0] for row in data]))
    second_column = list(map(str, [row[1] for row in data]))
    skip = False
    if qry_ref_name in second_column:
        if first_column[second_column.index(qry_ref_name)] == scene_name:
            skip = True
            print(f"skip pair {qry_ref_name} because it exist in line{int(second_column.index(qry_ref_name))+2}")
    return skip

def make_path(args, obj):
    out_dir = args.outdir
    obj_name = f'{obj:06d}'
    diff = f'gd_{args.limit_deg_min}_{args.limit_deg_max}'
    
    if args.use_pca_rgb == True:
        texture_type = 'pca_rgb'
    elif args.use_pca_msssim_loss == True:
        texture_type = 'pca'
    elif args.use_rgb_msssim_loss == True:
        texture_type = 'rgb'
    
    csv_path = os.path.join(out_dir, f'{args.bop_type}', 'results')
    os.makedirs(csv_path, exist_ok= True)
    video_path = os.path.join(out_dir, f'{args.bop_type}', obj_name, f'{texture_type}_{diff}')
    os.makedirs(video_path, exist_ok= True)
        
    return video_path, csv_path, texture_type 

def save_bop_results(path, results):

    lines = ['']
    
    lines.append('{scene_name},{im_id},{obj_id},{re_inital},{re_best},{R}'.format(
        scene_name = results['scene_name'],
        im_id = results['im_id'],
        obj_id = results['obj_id'],
        re_inital = results['re_inital'],
        re_best = results['re_best'],
        R =' '.join(map(str, results['R'].flatten().tolist())),)
        )

    with open(path, 'a') as f:
        f.write('\n'.join(lines))

def compute_column_mean(column_index, data):
    values = list(map(float, [row[column_index] for row in data]))
    total_values = len(values)
    mean = sum(values) / total_values
    return mean, values, total_values

def compute_csv_accuracy(file_path, total_result_path, obj_id):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        data.pop(0)
    
    lines = []

    mean_error, error_values, total_error_values = compute_column_mean(4, data)
    thresholds = [30, 15, 10, 5, 1]
    accuracies = [(sum(1 for value in error_values if value < threshold) / total_error_values) * 100 for threshold in thresholds]

    obj_name = f'instance_{obj_id:06d}'
    
    lines = ['\n{obj_name},{mean_error:.2f},{acc30:.2f},{acc15:.2f},{acc10:.2f},{acc5:.2f},{acc1:.2f},\n'.format(\
            obj_name=obj_name, mean_error=mean_error, acc30=accuracies[0], acc15=accuracies[1], acc10=accuracies[2], acc5=accuracies[3], acc1=accuracies[4])]
    
    with open(total_result_path, 'a') as f:
        f.write('\n'.join(lines))

def compute_init_csv_accuracy(file_path, total_result_path, obj_id):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        data.pop(0)
    
    lines = []
    erro_column_index = 3

    error_values = list(map(float, [row[erro_column_index] for row in data]))
    total_error_values = len(error_values)
    mean_error = sum(error_values) / total_error_values
    thresholds = [30, 15, 10, 5, 1]
    accuracies = [(sum(1 for value in error_values if value < threshold) / total_error_values) * 100 for threshold in thresholds]

    obj_name = f'instance_{obj_id:06d}'
    lines = ['\n{obj_name},{mean_error:.2f},{acc30:.2f},{acc15:.2f},{acc10:.2f},{acc5:.2f},{acc1:.2f}\n'.format(\
            obj_name=obj_name, mean_error=mean_error, acc30=accuracies[0], acc15=accuracies[1], acc10=accuracies[2], acc5=accuracies[3], acc1=accuracies[4])]
    
    with open(total_result_path, 'a') as f:
        f.write('\n'.join(lines))