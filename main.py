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
from utils import (setup_output, seed_all, save_bop_results, compute_csv_accuracy, preprocess_image, squeeze_mesh_info)
from libs.nvidiffrast import fit_pose

def main(args):     
    seed_all(929)             
    video_path, csv_path, texture_type, save_est_csv_path = setup_output(args)
    bop_dataset = BOP_Dataset(args, save_est_csv_path) if save_est_csv_path else BOP_Dataset(args, '')
    bop_dataloader = DataLoader(bop_dataset, batch_size=1, shuffle=True)
    
    glctx = dr.RasterizeCudaContext()
    
    for element in tqdm(bop_dataloader):
        ref_info, qry_info, mesh_info, pair_info = element
        pair_info = pair_info[0]
        mesh_info = squeeze_mesh_info(mesh_info, device='cuda')
        
        K = qry_info['K'].squeeze(0)
        R_mtx = qry_info['pose_rot'].squeeze(0).cuda()
        t = qry_info['t'].squeeze(0).cuda()
        
        qry_rgb = preprocess_image(qry_info['rgb']).cuda()
        qry_pca = preprocess_image(qry_info['pca']).cuda()
        
        pose_best_mtx, re_best, re_inital = fit_pose(
            args, glctx, K, R_mtx, t, mesh_info, qry_rgb, qry_pca,
        )
        # save results
        pred = dict(
            pair_info = pair_info,
            re_inital = re_inital,
            re_best = re_best,
            R = np.array(pose_best_mtx.clone().cpu()).reshape(3, 3),)
        
        save_bop_results(save_est_csv_path, pred)
    
    # compute total accuracy
    total_result_path = os.path.join(csv_path, f'{texture_type}_total.csv')
    total_result_path_init = os.path.join(csv_path, f'{texture_type}_total_init.csv')
    compute_csv_accuracy(save_est_csv_path, total_result_path, args.obj_id, column_index=2)
    compute_csv_accuracy(save_est_csv_path, total_result_path_init, args.obj_id, column_index=1)
    
if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    main(args)
    