import math
import numpy as np
import torch


#----------------------------------------------------------------------------
### inital pose sample
#----------------------------------------------------------------------------

def inner_product(a, b):
    return (a * b).sum(dim=-1)

def from_axis_angle(axis, angle):
    """
    Compute a quaternion from the axis angle representation.

    Reference:
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    Args:
        axis: axis to rotate about
        angle: angle to rotate by

    Returns:
        Tensor of shape (*, 4) representing a quaternion.
    """
    if torch.is_tensor(axis) and isinstance(angle, float):
        angle = torch.tensor(angle, dtype=axis.dtype, device=axis.device)
        angle = angle.expand(axis.shape[0])

    axis = axis / torch.norm(axis, dim=-1, keepdim=True)

    c = torch.cos(angle / 2.0)
    s = torch.sin(angle / 2.0)

    w = c
    x = s * axis[..., 0]
    y = s * axis[..., 1]
    z = s * axis[..., 2]

    return torch.stack((w, x, y, z), dim=-1)



@torch.jit.script
def normalize(vector, dim: int = -1):
    """
    Normalizes the vector to a unit vector using the p-norm.
    Args:
        vector (tensor): the vector to normalize of size (*, 3)
        p (int): the norm order to use

    Returns:
        (tensor): A unit normalized vector of size (*, 3)
    """
    return vector / torch.norm(vector, p=2.0, dim=dim, keepdim=True)

def uniform_unit_vector(n):
    return normalize(torch.randn(n, 3), dim=1)


@torch.jit.script
def ensure_batch_dim(tensor, num_dims: int):
    unsqueezed = False
    if len(tensor.shape) == num_dims:
        tensor = tensor.unsqueeze(0)
        unsqueezed = True

    return tensor, unsqueezed

def rotate_vector(quat, vector):
    """
    References:
            https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py#L419
    """
    assert quat.shape[-1] == 4
    assert vector.shape[-1] == 3
    assert quat.shape[:-1] == vector.shape[:-1]

    original_shape = list(vector.shape)
    quat = quat.view(-1, 4)
    vector = vector.view(-1, 3)

    pure_quat = quat[:, 1:]
    uv = torch.cross(pure_quat, vector, dim=1)
    uuv = torch.cross(pure_quat, uv, dim=1)
    return (vector + 2 * (quat[:, :1] * uv + uuv)).view(original_shape)

def evenly_distributed_points(n: int, hemisphere=False, pole=(0.0, 0.0, 1.0)):
    """
    Uses the sunflower method to sample points on a sphere that are
    roughly evenly distributed.

    Reference:
        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
    """
    indices = torch.arange(0, n, dtype=torch.float32) + 0.5

    if hemisphere:
        phi = torch.acos(1 - 2 * indices / n / 2)
    else:
        phi = torch.acos(1 - 2 * indices / n)
    theta = math.pi * (1 + 5 ** 0.5) * indices

    points = torch.stack([
        torch.cos(theta) * torch.sin(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(phi),
    ], dim=1)

    if hemisphere:
        default_pole = torch.tensor([(0.0, 0.0, 1.0)]).expand(n, 3)
        pole = torch.tensor([pole]).expand(n, 3)
        if (default_pole[0] + pole[0]).abs().sum() < 1e-5:
            # If the pole is the opposite side just flip.
            points = -points
        elif (default_pole[0] - pole[0]).abs().sum() < 1e-5:
            points = points
        else:
            # Otherwise take the cross product as the rotation axis.
            rot_axis = torch.cross(pole, default_pole)
            rot_angle = torch.acos(inner_product(pole, default_pole))
            rot_quat = from_axis_angle(rot_axis, rot_angle)
            points = rotate_vector(rot_quat, points)

    return points

def evenly_mat_from_ray(forward, up=None):
    """
    Quaternions that orients the camera forward direction.

    Args:
        forward: a vector representing the forward direction.
    """
    n = forward.shape[0]
    up = torch.tensor(up).unsqueeze(0).expand(n, 3)
    up = up + forward
    down = -up
    right = normalize(torch.cross(down, forward))
    down = normalize(torch.cross(forward, right))

    mat = torch.stack([right, down, forward], dim=1)

    return mat

def evenly_down_vector(num_samples):
    """
    Sample evenly random down vectors.

    Args:
        num_samples: the number of sampled down vectors
    """

    samples = []
    for i in range(num_samples):
        theta = 2 * np.pi * i / num_samples
        phi = np.pi / 2  
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        samples.append(np.array([x, y, z]))
    samples = np.stack(samples)
    return torch.tensor(samples, dtype=torch.float32)

def evenly_n_mat_from_ray(forward, n = None, up=None):
    """
    Sample evenly quaternions that orients the camera forward direction.

    Args:
        forward: a vector representing the forward direction.
    """
    n_viewpoint = forward.shape[0]
    n_inpalne_rotation = n * n_viewpoint
    forward = forward.tile((n, 1))
    evenly_down = evenly_down_vector(n_inpalne_rotation)
    evenly_right = normalize(torch.cross(evenly_down, forward))
    evenly_down = normalize(torch.cross(forward, evenly_right))
    mat_0 = torch.stack([evenly_right, evenly_down, forward], dim=1)
    return mat_0

def evenly_distributed_mats(viewpoint: int, 
                            inplane_rotation: int, 
                            hemisphere=False, 
                            hemisphere_pole=(0.0, 0.0, 1.0),
                            upright_up=(0.0, 0.0, 1.0)):
    """
    Sample evenly distributed rotation mats.

    Args:
        viewpoint: number of sampled viewpoints.
        inplane_rotation: number of sampled inplane_rotations.
        hemisphere: sample in hemisphere.
        upright_up: up vector.
    """
    rays = evenly_distributed_points(viewpoint, hemisphere, hemisphere_pole)
    total_mat = torch.cat((evenly_mat_from_ray(-rays, upright_up), \
        evenly_n_mat_from_ray(-rays, inplane_rotation)), dim = 0)
    
    return total_mat