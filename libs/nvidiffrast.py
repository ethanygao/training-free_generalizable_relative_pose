import os, sys
CUR_dir = os.path.abspath(os.path.dirname(__file__))
ROOT_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_dir)
sys.path.append(CUR_dir)

import math
import torch
import numpy as np
import nvdiffrast.torch as dr
from libs.loss import MS_SSIM_CUDA
from libs.create_pose import gen_init_pose
from utils import (quaternion_to_rot_mat, matrix_to_quaternion, rotation_error)
ms_ssim_func = MS_SSIM_CUDA(data_range=1.0, normalize=True).cuda()

def trans_pts(pts, R, t, K, resolution):
    if len(R.shape) == 2:
        R = R[None, ...]
    if len(t.shape) == 2:
        t = t[None, ...]
    if len(K.shape) == 3:
        K = K[0]
        
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

def render(glctx, R, t, K, pos, pos_idx, col, col_idx, resolution: tuple, use_backface_culling):
    pos_clip    =  trans_pts(pos, R, t, K, resolution) # or minibatch
    
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolution, backface_culling = use_backface_culling)
    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    # color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    
    return color

def render_initial(glctx, Rs, ts, K, mesh_info, resolution, qry_rgb, qry_pca, \
    use_backface_culling, use_rgb_msssim_loss, use_pca_msssim_loss):
    '''render images according to sampled rotations and choose the one that minimizes loss'''
    bs = Rs.shape[0]
    rgb_pos_idx, rgb_vtx_pos, rgb_vtx_col, pca_vtx_col = mesh_info

    rgb_opt = render(glctx, Rs, ts, K, rgb_vtx_pos, rgb_pos_idx, rgb_vtx_col, rgb_pos_idx, resolution, use_backface_culling)
    pca_opt = render(glctx, Rs, ts, K, rgb_vtx_pos, rgb_pos_idx, pca_vtx_col, rgb_pos_idx, resolution, use_backface_culling)
    
    with torch.amp.autocast(device_type='cuda', enabled=True):  
        if use_rgb_msssim_loss == True:
            inp_rgb_exp = qry_rgb.permute(0,3,1,2).expand(bs, -1, -1, -1) # torch.Size([bs, 3, 480, 640])
            loss_rgb = 1 - ms_ssim_func(rgb_opt.permute(0,3,1,2).contiguous(), inp_rgb_exp.contiguous())
        else:
            loss_rgb = 0

        if use_pca_msssim_loss == True:
            inp_pca_exp = qry_pca.permute(0,3,1,2).expand(bs, -1, -1, -1)
            loss_pca = 1 - ms_ssim_func(pca_opt.permute(0,3,1,2).contiguous(), inp_pca_exp.contiguous())
        else:
            loss_pca = 0

    loss_init = loss_rgb + loss_pca
    min_loss, min_indices = torch.min(loss_init, dim=0)
    
    init_R_mtx = Rs[min_indices]
    return min_loss, init_R_mtx

def choose_best_rotation(args, glctx, pose_t, K, mesh_info, resolution, qry_rgb, qry_pca):
    Rs, ts = gen_init_pose(pose_t, args.viewpoint, args.inplane_rotation, args.hemisphere)
    min_loss = np.inf
    
    chunks = math.ceil(Rs.shape[0]/100)
    Rs_chunks = torch.chunk(Rs, chunks=chunks, dim=0)
    ts_chunks = torch.chunk(ts, chunks=chunks, dim=0)
    
    with torch.no_grad():
        for Rs_chunk, ts_chunk in zip(Rs_chunks, ts_chunks):            
            cur_loss, cur_R_mtx = render_initial(glctx, Rs_chunk, ts_chunk, K, mesh_info, resolution, qry_rgb, qry_pca, \
                args.use_backface_culling, args.use_rgb_msssim_loss, args.use_pca_msssim_loss)
            if cur_loss < min_loss:
                min_loss = cur_loss
                init_R_mtx = cur_R_mtx
    
    return init_R_mtx


def fit_pose(args, glctx, K, tgt_R_mtx, pose_t, mesh_info, qry_rgb, qry_pca):
    rgb_pos_idx, rgb_vtx_pos, rgb_vtx_col, pca_vtx_col = mesh_info
    resolution=(args.height,args.width)
    
    init_R_mtx = choose_best_rotation(args, glctx, pose_t, K, mesh_info, (args.height,args.width), qry_rgb, qry_pca)
    init_R_quat = matrix_to_quaternion(init_R_mtx)
    re_inital = rotation_error(init_R_mtx, tgt_R_mtx)  
    
    opt_R_quat    = (init_R_quat / torch.sum(init_R_quat**2)**0.5).type(torch.float32).cuda()
    opt_R_quat.requires_grad=True
    best_R_quat   = opt_R_quat.detach().clone()

    optimizer = torch.optim.Adam([opt_R_quat], betas=(0.9, 0.999), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=args.lr_patience_num, # 0.5*max_iter
            verbose=False,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=args.lr_patience_num,
            min_lr=1e-3,
            eps=1e-7)
    
    loss_best   = np.inf

    for it in range(args.max_iter + 1):        
        opt_R_mtx = quaternion_to_rot_mat(opt_R_quat)
        # Render.
        rgb_opt = render(glctx, opt_R_mtx, pose_t, K, rgb_vtx_pos, rgb_pos_idx, rgb_vtx_col, rgb_pos_idx, resolution, args.use_backface_culling)
        pca_opt   = render(glctx, opt_R_mtx, pose_t, K, rgb_vtx_pos, rgb_pos_idx, pca_vtx_col, rgb_pos_idx, resolution, args.use_backface_culling)
        
        if args.use_rgb_msssim_loss == True:
            loss_rgb = 1 - ms_ssim_func(rgb_opt.permute(0,3,1,2).contiguous(), qry_rgb.permute(0,3,1,2).contiguous())
        else:
            loss_rgb = 0 # torch.mean((rgb_opt - inp_rgb)**2)
        if args.use_pca_msssim_loss == True:
            loss_pca = 1 - ms_ssim_func(pca_opt.permute(0,3,1,2).contiguous(), qry_pca.permute(0,3,1,2).contiguous())
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
        if args.log_interval and (it % args.log_interval == 0):
            for param_group in optimizer.param_groups:
                lr_current = param_group['lr']
            s = "iter=%d,re=%f,re_best=%f,loss=%f,loss_best=%f,lr=%f" % (it, re, re_best, loss_val, loss_best, lr_current)
            print(s)
        
        # Run gradient training step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.clone()) 

        with torch.no_grad():
            opt_R_quat /= torch.sum(opt_R_quat**2)**0.5
            
    return pose_best_mtx, re_best, re_inital
