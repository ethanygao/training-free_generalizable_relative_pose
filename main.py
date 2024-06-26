import os,sys
ROOT_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_dir)

import torch
import numpy as np
from tqdm import tqdm
import nvdiffrast.torch as dr
from torch.utils.data import DataLoader
from config import config_parser
from data.bop_dataset import BOP_Dataset
from libs.nvidiffrast import fit_pose
from utils import make_path, check_exists, save_bop_results, compute_csv_accuracy, compute_init_csv_accuracy

def preprocess_image(image):
    image = image.squeeze().float() / 255.0
    image = image.flip(0).unsqueeze(0)
    return image

def main(args):            
    bop_dataset = BOP_Dataset(args)
    bop_dataloader = DataLoader(dataset = bop_dataset, batch_size=1, shuffle=True)

    for obj in args.obj_id_list:
        if args.outdir is not None:
            video_path, csv_path, texture_type = make_path(args, obj)

    with open(f'{csv_path}/config.txt', 'w') as file:
        print(f'{csv_path}/config.txt')
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
            
    glctx= dr.RasterizeCudaContext()
    
    for element in tqdm(bop_dataloader):
        ref_info, qry_info  = element
        obj_id = qry_info['obj_id'].squeeze().item()
        
        qry_scene_id, ref_scene_id = qry_info['scene_id'].squeeze().item(), ref_info['scene_id'].squeeze().item()
        qry_im_id, ref_im_id = qry_info['im_id'].squeeze().item(), ref_info['im_id'].squeeze().item()
        
        scene_name = f'{qry_scene_id}_{ref_scene_id}'
        ref_qry_name = f'{qry_im_id}_{ref_im_id}'
        
        save_est_csv_path = os.path.join(csv_path, f'{texture_type}_est_{obj_id:06d}.csv')
        if os.path.exists(save_est_csv_path):
            skip = check_exists(save_est_csv_path, ref_qry_name, str(scene_name))
            if skip == True:
                continue
        
        qry_rgb, ref_rgb  = qry_info['rgb'] , ref_info['rgb']
        qry_pca, ref_pca  = qry_info['pca'], ref_info['pca']

        qry_K = qry_info['K'].squeeze(0)
        qry_pose_rot = qry_info['pose_rot'].squeeze(0)
        pose_t = qry_info['pose_t'].squeeze(0)
        
        rgb_mesh_path = ref_info['rgb_mesh_path'][0]
        pca_mesh_path = ref_info['pca_mesh_path'][0]
    
        qry_im_rgb = preprocess_image(qry_rgb)
        qry_im_pca = preprocess_image(qry_pca)
        ref_im_rgb = preprocess_image(ref_rgb)
        ref_im_pca = preprocess_image(ref_pca)
        
        mp4save_fn = f'{qry_scene_id}_{ref_scene_id}_{qry_im_id}_{ref_im_id}.mp4'
        log_fn = f'log_{qry_scene_id}_{ref_scene_id}_{qry_im_id}_{ref_im_id}.txt'
        
        pred_rot_mtx, re_best, re_inital = fit_pose(
                glctx = glctx,
                rgb_mesh_path = rgb_mesh_path, 
                pca_mesh_path = pca_mesh_path, 
                tgt_R_mtx = qry_pose_rot,
                t = pose_t,
                K     = qry_K,
                inp_rgb = qry_im_rgb,
                inp_pca = qry_im_pca,
                ref_rgb = ref_im_rgb,
                ref_pca = ref_im_pca,
                max_iter = args.max_iter,
                resolution = (args.height,args.width),
                lr_base = args.lr,
                lr_patience_num = args.lr_patience_num,
                log_interval = 10,
                out_dir = video_path,
                log_fn = log_fn,
                mp4save_interval = args.mp4save_interval,
                mp4save_fn = mp4save_fn,
                use_backface_culling = args.use_backface_culling,
                use_rgb_msssim_loss = args.use_rgb_msssim_loss, 
                use_pca_msssim_loss = args.use_pca_msssim_loss,
                noise_num = args.noise_num,
                viewpoint = args.viewpoint,
                hemisphere = args.hemisphere,
                inplane_rotation = args.inplane_rotation)

        # save results
        pred = dict(
            scene_name = scene_name,
            im_id = f'{qry_im_id}_{ref_im_id}',
            obj_id = obj_id,
            re_inital = re_inital,
            re_best = re_best,
            R = np.array(pred_rot_mtx.clone().cpu()).reshape(3, 3),)
        
        save_bop_results(save_est_csv_path, pred)
    
    # compute total accuracy
    total_result_path = os.path.join(csv_path, f'{texture_type}_total.csv')
    total_result_path_init = os.path.join(csv_path, f'{texture_type}_total_init.csv')
    compute_csv_accuracy(save_est_csv_path, total_result_path, obj_id)
    compute_init_csv_accuracy(save_est_csv_path, total_result_path_init, obj_id)
    
if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    main(args)
    