import os, sys
CUR_dir = os.path.abspath(os.path.dirname(__file__))
ROOT_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_dir)
sys.path.append(CUR_dir)

import numpy as np
import torch
import torch.nn.functional as F
import math
from PIL import Image
import nvdiffrast.torch as dr
from libs.loss import MS_SSIM
from create_pose import evenly_distributed_mats
from utils import *

ms_ssim_func = MS_SSIM(data_range=1.0, normalize=True).cuda()

def gen_init_pose(t, viewpoint=100, inplane_rotation=3, hemisphere = False):    
    rs = evenly_distributed_mats(viewpoint, inplane_rotation, hemisphere = hemisphere).cuda()
    ts = t.expand(rs.shape[0], -1, -1)
    return rs, ts

def gen_noise_pose(init_R_mtx, t, iter = 300):
    sampled_nrs = list(np.linspace(0, 1, iter))
    init_quat = matrix_to_quaternion(init_R_mtx)
    noise_R_mtx_batch = []
    for nr in sampled_nrs:
        noise = q_scale(q_rnd(), nr)
        noise = q_mul(noise, q_rnd_S4())         
        total_opt_R_quat = q_mul_torch(init_quat, noise)
        noise_R_mtx = quaternion_to_rot_mat(total_opt_R_quat)
        noise_R_mtx_batch.append(noise_R_mtx)
    
    Rs_noise = torch.stack(noise_R_mtx_batch, axis=0)
    ts = t.expand(Rs_noise.shape[0], -1, -1)
    
    return Rs_noise, ts

def load_mesh_data(mesh_path):
    mesh_data = load_ply_mesh(mesh_path)
    pos = mesh_data["v"]
    pos_idx = mesh_data["f"]
    col = mesh_data["colors"]
    col_idx = mesh_data["f"]
    
    if pos.shape[1] == 4:
        pos = pos[:, 0:3]
        
    pos_idx = pos_idx.type(torch.int32).cuda()
    vtx_pos = pos.type(torch.float32).cuda()
    col_idx = col_idx.type(torch.int32).cuda()
    vtx_col = col.type(torch.float32).cuda()
    
    return pos_idx, vtx_pos, col_idx, vtx_col

def load_rgb_and_pca_meshes(rgb_mesh_path, pca_mesh_path):
    rgb_pos_idx, rgb_vtx_pos, rgb_col_idx, rgb_vtx_col = load_mesh_data(rgb_mesh_path)
    pca_pos_idx, pca_vtx_pos, pca_col_idx, pca_vtx_col = load_mesh_data(pca_mesh_path)
    
    return (rgb_pos_idx, rgb_vtx_pos, rgb_col_idx, rgb_vtx_col, \
           pca_pos_idx, pca_vtx_pos, pca_col_idx, pca_vtx_col)

def trans_pts_batch(pts, R, t, K, resolution):
    resolution = torch.tensor(resolution, dtype=torch.float32)
    new_positions  = torch.matmul(R, pts.t()) + t
    new_positions = new_positions.permute(0,2,1).contiguous()

    depth_scale = torch.max(new_positions[:, :, 2])
    d = new_positions[:, :, 2] / depth_scale
    u = (new_positions[:, :, 0] * K[0, 0] / new_positions[:, :, 2]) + K[0, 2]
    v = (new_positions[:, :, 1] * K[1, 1] / new_positions[:, :, 2]) + K[1, 2]

    u = ( u * 2.0 / resolution[1]) - 1.0
    v = (-v * 2.0 / resolution[0]) + 1.0
    
    gl_Position = torch.stack([u, v, d , torch.ones_like(d)], dim=2)
    
    return gl_Position

def render_batch(glctx, R, t, K, pos, pos_idx, col, col_idx, resolution: tuple, use_backface_culling):
    pos_clip    =  trans_pts_batch(pos, R, t, K, resolution) # or minibatch
    
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolution, backface_culling = use_backface_culling)
    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    # color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    
    return color

def trans_pts(pts, R, t, K, resolution):
    resolution = torch.tensor(resolution, dtype=torch.float32)
    new_positions  = torch.matmul(R, pts.t()) + t
    new_positions = new_positions.t().contiguous()

    depth_scale = torch.max(new_positions[:, 2])
    d = new_positions[:, 2] / depth_scale
    u = (new_positions[:, 0] * K[0, 0] / new_positions[:, 2]) + K[0, 2]
    v = (new_positions[:, 1] * K[1, 1] / new_positions[:, 2]) + K[1, 2]

    u = ( u * 2.0 / resolution[1]) - 1.0
    v = (-v * 2.0 / resolution[0]) + 1.0
    
    gl_Position = torch.stack([u, v, d , torch.ones_like(d)], dim=1)
    
    return gl_Position[None,...]

def render(glctx, R, t, K, pos, pos_idx, col, col_idx, resolution: tuple, use_backface_culling):
    pos_clip    =  trans_pts(pos, R, t, K, resolution)
    
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolution, backface_culling = use_backface_culling)
    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    # color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    
    return color


def choose_best_rotation(glctx, Rs, ts, K, mesh_info, resolution, inp_rgb, inp_pca, \
    use_backface_culling, use_rgb_msssim_loss, use_pca_msssim_loss):
    '''render images according to sampled rotations and choose the one that minimizes loss'''
    
    bs = Rs.shape[0]
    rgb_pos_idx, rgb_vtx_pos, rgb_col_idx, rgb_vtx_col, \
        pca_pos_idx, pca_vtx_pos, pca_col_idx, pca_vtx_col = mesh_info

    rgb_opt = render_batch(glctx, Rs, ts, K, rgb_vtx_pos, rgb_pos_idx, rgb_vtx_col, rgb_col_idx, resolution, use_backface_culling)
    pca_opt = render_batch(glctx, Rs, ts, K, pca_vtx_pos, pca_pos_idx, pca_vtx_col, pca_col_idx, resolution, use_backface_culling)
    
    if use_rgb_msssim_loss == True:
        inp_rgb_exp = inp_rgb.permute(0,3,1,2).expand(bs, -1, -1, -1) # torch.Size([bs, 3, 480, 640])
        loss_rgb = 1 - ms_ssim_func(rgb_opt.permute(0,3,1,2), inp_rgb_exp)
    else:
        loss_rgb = 0

    if use_pca_msssim_loss == True:
        inp_pca_exp = inp_pca.permute(0,3,1,2).expand(bs, -1, -1, -1)
        loss_pca = 1 - ms_ssim_func(pca_opt.permute(0,3,1,2), inp_pca_exp)
    else:
        loss_pca = 0

    loss_init = loss_rgb + loss_pca
    min_loss, min_indices = torch.min(loss_init, dim=0)
    
    init_R_mtx = Rs[min_indices]
    
    return min_loss, init_R_mtx
    
#----------------------------------------------------------------------------
# pose fitter.
#----------------------------------------------------------------------------

def fit_pose(
            bop_type = None,
            obj_id = None,
            ref_rgb_crop = None,
            qry_rgb_crop = None,
            glctx              = None,
            rgb_mesh_path      = None,
            pca_mesh_path      = None,
            tgt_R_mtx          = None,
            t                  = None,
            K                  = None,
            inp_rgb            = None,
            inp_pca            = None,
            ref_rgb            = None,
            ref_pca            = None,
            max_iter           = 10000,
            r_level            = 1,
            lr_patience_num    = None,
            log_interval       = 5,
            lr_base            = 0.001,
            resolution         = (192,256),
            out_dir            = None,
            log_fn             = None,
            mp4save_interval   = None,
            mp4save_fn         = None,
            hemisphere = None, 
            use_backface_culling = None,
            use_rgb_msssim_loss =True, 
            use_pca_msssim_loss =True, 
            noise_num = None,
            viewpoint = None,
            inplane_rotation = None):
    
    '''log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(out_dir + '/' + log_fn, 'wt')
        if mp4save_interval != 0:
            writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', fps=5, codec='libx264', bitrate='16M')
        else:
            mp4save_interval = None'''

    mesh_info = load_rgb_and_pca_meshes(rgb_mesh_path, pca_mesh_path)
    
    rgb_pos_idx, rgb_vtx_pos, rgb_col_idx, rgb_vtx_col, \
        pca_pos_idx, pca_vtx_pos, pca_col_idx, pca_vtx_col = mesh_info
    
    Rs, ts = gen_init_pose(t, viewpoint, inplane_rotation, hemisphere)
    min_loss = np.inf
    
    chunks = math.ceil(Rs.shape[0]/100)
    Rs_chunks = torch.chunk(Rs, chunks=chunks, dim=0)
    ts_chunks = torch.chunk(ts, chunks=chunks, dim=0)
    
    for Rs_chunk, ts_chunk in zip(Rs_chunks, ts_chunks):            
        cur_loss, cur_R_mtx = choose_best_rotation(glctx, Rs_chunk, ts_chunk, K, mesh_info, resolution, inp_rgb, inp_pca, \
            use_backface_culling, use_rgb_msssim_loss, use_pca_msssim_loss)
        if cur_loss < min_loss:
            min_loss = cur_loss
            init_R_mtx = cur_R_mtx
            
    re_inital = rotation_error(init_R_mtx, tgt_R_mtx)
    re_inital_log = f'first_initial_error: {re_inital}, min_loss: {min_loss.item()}'

    '''if log_file:
        log_file.write(re_inital_log + "\n")'''
        
    init_R_quat = matrix_to_quaternion(init_R_mtx)
    opt_R_quat    = (init_R_quat / torch.sum(init_R_quat**2)**0.5).type(torch.float32).cuda()
    opt_R_quat.requires_grad=True
    best_R_quat   = opt_R_quat.detach().clone()

    loss_best   = np.inf
    optimizer = torch.optim.Adam([opt_R_quat], betas=(0.9, 0.999), lr=lr_base)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=lr_patience_num, # 0.5*max_iter
            verbose=False,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=lr_patience_num,
            min_lr=1e-3,
            eps=1e-7)
        
    if max_iter == 0:
        re_best = rotation_error(init_R_mtx,tgt_R_mtx)
        return init_R_mtx, re_best, re_inital
    
    for it in range(max_iter + 1):        
        opt_R_mtx = quaternion_to_rot_mat(opt_R_quat)
        # Render.
        rgb_opt = render(glctx, opt_R_mtx, t, K, rgb_vtx_pos, rgb_pos_idx, rgb_vtx_col, rgb_col_idx, resolution, use_backface_culling)
        pca_opt   = render(glctx, opt_R_mtx, t, K, pca_vtx_pos, pca_pos_idx, pca_vtx_col, pca_col_idx, resolution, use_backface_culling)
        
        if use_rgb_msssim_loss == True:
            loss_rgb = 1 - ms_ssim_func(rgb_opt.permute(0,3,1,2), inp_rgb.permute(0,3,1,2))
        else:
            loss_rgb = 0 # torch.mean((rgb_opt - inp_rgb)**2)
        if use_pca_msssim_loss == True:
            loss_pca = 1 - ms_ssim_func(pca_opt.permute(0,3,1,2), inp_pca.permute(0,3,1,2))
        else:
            loss_pca = 0 # torch.mean((pca_opt - inp_pca)**2)

        loss = loss_rgb + loss_pca
        loss_val = float(loss)
        
        # Measure image-space loss and update best found pose.
        if (loss_val < loss_best) and (loss_val > 0.0):    
            best_R_quat = opt_R_quat.detach().clone()
            loss_best = loss_val
        
        pose_opt_mtx = quaternion_to_rot_mat(opt_R_quat)
        pose_best_mtx = quaternion_to_rot_mat(best_R_quat)
        
        re = rotation_error(pose_opt_mtx,tgt_R_mtx)
        re_best = rotation_error(pose_best_mtx,tgt_R_mtx)
            
        # Print/save log.
        if log_interval and (it % log_interval == 0):
            for param_group in optimizer.param_groups:
                lr_current = param_group['lr']
            s = "iter=%d,re=%f,re_best=%f,loss=%f,loss_best=%f,lr=%f" % (it, re, re_best, loss_val, loss_best, lr_current)
            print(s)
            '''if log_file:
                log_file.write(s + "\n")'''

        # Run gradient training step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.clone()) 

        with torch.no_grad():
            opt_R_quat /= torch.sum(opt_R_quat**2)**0.5
                
        ''' save_mp4  = mp4save_interval and (it % mp4save_interval == 0)
        if save_mp4:
            qry_color_rgb, ref_color_rgb, qry_rgb_opt = [img[0].clone().detach().cpu().numpy() for img in [inp_rgb, ref_rgb, rgb_opt]]
            qry_color_pca, ref_color_pca, img_pca_opt = [img[0].clone().detach().cpu().numpy() for img in [inp_pca, ref_pca, pca_opt]]

            # reference image / query image / renderd image
            result_rgb_image = np.concatenate([ref_color_rgb, qry_color_rgb, qry_rgb_opt], axis=1)[::-1]
            result_pca_image = np.concatenate([ref_color_pca, qry_color_pca, img_pca_opt], axis=1)[::-1]
            result_image = np.concatenate([result_rgb_image, result_pca_image], axis=0)
            writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))

    # Done.
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()'''
        
    return pose_best_mtx, re_best, re_inital

