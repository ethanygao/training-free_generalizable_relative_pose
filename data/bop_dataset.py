import os,sys
CUR_dir = os.path.abspath(os.path.dirname(__file__))
ROOT_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_dir)
sys.path.append(CUR_dir)

import cv2
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from config import config_parser
from data.gen_pairs import gen_pairs, gen_ycbv_pairs
from libs.extractor_dino import build_dino_extractor, get_pca_image_dinov2
from libs.recon_mesh import mesh_from_depth
from utils import get_image_crop_resize, get_K_crop_resize

class BOP_Dataset(Dataset):
    def __init__(self, args, csv_path=''):
        self.args = args
        self.dino_modle  = build_dino_extractor('large')
        self.scale_depth_factor = 0.1 if args.bop_type == 'ycbv' or args.bop_type == 'tless' else 1.0
 
        self.file_name = 'test'
        
        self.ds_dir = os.path.join(args.ds_dir, args.bop_type)
        self.depth_name = 'gt'
        self.base_dir = os.path.join(self.ds_dir, self.file_name)
        
        self.obj_id = args.obj_id
        pair_root = os.path.join(ROOT_dir, f"data/gd_pairs/{self.args.bop_type}")
        os.makedirs(pair_root, exist_ok= True)
        pair_path = f"{pair_root}/{self.obj_id:06d}.json"

        if not os.path.exists(pair_path):
            if args.bop_type == 'lm' or args.bop_type =='lmo':
                gen_pairs(args, pair_path, self.obj_id)   
            elif args.bop_type == 'ycbv':
                gen_ycbv_pairs(args, pair_path, self.obj_id, file_name = self.file_name, num_pairs= 20000)   
        
        with open(pair_path) as f:
            pair_list = json.load(f)
            pair_list = pair_list[str(self.obj_id)]
            
        if os.path.isfile(csv_path):
            with open(os.path.join(csv_path), 'r') as file:
                reader = csv.reader(file)
                data = list(reader)
                data.pop(0)
                exist_pair_list = list(map(str, [f"{row[0].split('_')[2]}_{row[0].split('_')[3]}" for row in data]))
                exist_pair_list.sort()
            rest_pair_list = []
            for pair in pair_list:
                parts = pair.split('_')
                if len(parts) >= 2:
                    prefix = '_'.join(parts[:2])
                    if prefix in exist_pair_list:
                        continue
                rest_pair_list.append(pair)
    
            self.pair_list = rest_pair_list    
        else:
            self.pair_list = pair_list     
            
        self.scene_info = self._load_all_scene_metadata()
        self.obj_id_mapping = self._build_obj_id_mapping()
    
    def _load_all_scene_metadata(self):
        scene_info = {}
        for scene_dir in os.listdir(self.base_dir):
            scene_path = os.path.join(self.base_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            scene_id = int(scene_dir)
            metadata = {
                'gt_info': json.load(open(os.path.join(scene_path, 'scene_gt_info.json'))),
                'camera': json.load(open(os.path.join(scene_path, 'scene_camera.json'))),
                'gt': json.load(open(os.path.join(scene_path, 'scene_gt.json')))
            }
            scene_info[scene_id] = metadata
        return scene_info
    
    def _build_obj_id_mapping(self):
        mapping = {}
        for scene_id, metadata in self.scene_info.items():
            for img_id_str in metadata['gt']:
                img_id = int(img_id_str)
                for obj_idx, obj in enumerate(metadata['gt'][img_id_str]):
                    if obj['obj_id'] == self.obj_id:
                        mapping[(scene_id, img_id)] = obj_idx
                        break
        return mapping
    
    def get_crop_info(self, K, bbox, resize_shape, rgb, mask, depth):
        K_crop, K_crop_homo = get_K_crop_resize(bbox, K, resize_shape)
        rgb_crop, _ = get_image_crop_resize(rgb, bbox, resize_shape)
        mask_crop, _ = get_image_crop_resize(mask, bbox, resize_shape)
        depth_crop, _ = get_image_crop_resize(depth, bbox, resize_shape)
        
        mask_crop_ = np.zeros_like(mask_crop)
        mask_crop_[mask_crop > 0] = 255
        
        return rgb_crop, mask_crop_, depth_crop, K_crop, K_crop_homo
    
    def get_crop_images(self, rgb, mask, depth, bbox, K, compact_percent = 0.5):
        x0, y0, w, h = bbox
        x1, y1 = x0+w, y0+h
        s = max(w, h)
        x0 -= int(s * compact_percent)
        y0 -= int(s * compact_percent)
        x1 += int(s * compact_percent)
        y1 += int(s * compact_percent)
        
        box_1 = np.array([x0, y0, x1, y1])
        s_shape_1 = max(y1 - y0, x1 - x0)
        resize_shape_1 = np.array([s_shape_1, s_shape_1])
        rgb_crop_1, mask_crop_1, depth_crop_1, K_crop_1, _ = \
            self.get_crop_info(K, box_1, resize_shape_1, rgb, mask, depth)
        
        box_2 = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape_2 = np.array([self.args.height, self.args.width]) # np.array([192,256])
        rgb_crop_2, mask_crop_2, depth_crop_2, K_crop_2, _ = \
            self.get_crop_info(K_crop_1, box_2, resize_shape_2, rgb_crop_1, mask_crop_1, depth_crop_1) 
            
        return rgb_crop_1, mask_crop_1, depth_crop_1, K_crop_1, \
            rgb_crop_2, mask_crop_2, depth_crop_2, K_crop_2, box_2, resize_shape_2
            
    
    def get_img_info(self, base_dir, scene_id, img_id, is_ref = True, ref_projection = None):
        if is_ref == False:
            assert(ref_projection is not None)
        
        rgb_path = os.path.join(base_dir, f'{scene_id:06d}', 'rgb', f'{img_id:06d}.png')
        depth_path = rgb_path.replace('rgb','depth')

        meta = self.scene_info[scene_id]
        obj_idx = self.obj_id_mapping[(scene_id, img_id)]
        mask_visib_path = os.path.join(base_dir, f'{scene_id:06d}', 'mask_visib',f'{img_id:06d}_{obj_idx:06d}.png')
        
        bbox_visib = meta['gt_info'][str(img_id)][obj_idx]['bbox_visib']
        K = np.array(meta['camera'][str(img_id)]['cam_K'], dtype=np.float32).reshape(3,3)
        R = np.array(meta['gt'][str(img_id)][obj_idx]['cam_R_m2c'], dtype=np.float32).reshape(3,3)
        t = np.array(meta['gt'][str(img_id)][obj_idx]['cam_t_m2c'], dtype=np.float32).reshape(3,1)
        
        RT_m2c = np.concatenate([np.concatenate([R, t], axis=1), np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0).astype(np.float32)
        pose_inv = np.linalg.inv(RT_m2c)              
                
        rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32)
        mask = np.array(Image.open(mask_visib_path))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) 
        
        rgb_crop_1, mask_crop_1, depth_crop_1, K_crop_1,\
            rgb_crop_2, mask_crop_2, depth_crop_2, K_crop_2, box_2, resize_shape_2 = self.get_crop_images(rgb, mask, depth, bbox_visib, K)
        
        rgb_mask_1 = (rgb_crop_1 * mask_crop_1[:,:,None] / 255).astype(np.float32)
        rgb_mask_2 = (rgb_crop_2 * mask_crop_2[:,:,None] / 255).astype(np.float32)
        
        rgb_mask_pil_1 = Image.fromarray(rgb_mask_1.astype(np.uint8))
        depth_mask_1 = depth_crop_1 * (mask_crop_1 / 255 * self.scale_depth_factor).astype(np.float32)
                
        projection = None
        recon_info = ()
        
        if is_ref == True:
            pca_crop_1, projection = get_pca_image_dinov2(self.dino_modle, rgb_mask_pil_1, mask_crop_1, is_ref = True)  
        else:
            pca_crop_1 = get_pca_image_dinov2(self.dino_modle, rgb_mask_pil_1, mask_crop_1, is_ref = False, projection = ref_projection)

        pca_mask_1 = pca_crop_1 * (mask_crop_1[:, : , None] / 255).astype(np.float32)
        pca_mask_2, _ = get_image_crop_resize(pca_mask_1, box_2, resize_shape_2) 
        pca_mask_2 = pca_mask_2.astype(np.float32)
        
        if is_ref == True:
            recon_info = (depth_mask_1, rgb_mask_1, pca_mask_1, K_crop_1, pose_inv)
                
        obs = dict(
                scene_id = torch.tensor(scene_id),
                obj_id = torch.tensor(self.obj_id), 
                im_id = torch.tensor(img_id),
                K = torch.tensor(K_crop_2),
                rgb = torch.tensor(rgb_mask_2),
                pca = torch.tensor(pca_mask_2),
                pose_rot = torch.tensor(R),
                t = torch.tensor(t),
                recon_info = recon_info,)
        
        return obs, projection
    
    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, pair_idx):

        pair_name = self.pair_list[pair_idx]
        base_name = os.path.basename(pair_name)
        
        if self.args.bop_type == 'ycbv':
            qry_scene_id, qry_idx, ref_scene_id, ref_idx  = \
            int(base_name.split("_")[1]), int(base_name.split("_")[2]),\
                int(base_name.split("_")[4]), int(base_name.split("_")[5])
        elif self.args.bop_type == 'lm':
            qry_idx, ref_idx = int(base_name.split("_")[0]), int(base_name.split("_")[1])
            qry_scene_id = ref_scene_id = self.obj_id
        elif self.args.bop_type == 'lmo':
            qry_idx, ref_idx = int(base_name.split("_")[0]), int(base_name.split("_")[1])
            qry_scene_id = ref_scene_id = 2
        
        ref_info, re_projection = self.get_img_info(self.base_dir, ref_scene_id, ref_idx, is_ref=True)
        qry_info, _             = self.get_img_info(self.base_dir, qry_scene_id, qry_idx, is_ref=False, ref_projection = re_projection)
        
        max_depth_diff_ratio = 10 if self.args.bop_type == 'ycbv' else 4
        depth_mask_1, rgb_mask_1, pca_mask_1, K_crop_1, pose_inv = ref_info['recon_info']
        
        rgb_vtx_pos, rgb_pos_idx, rgb_vtx_col, pca_vtx_col = mesh_from_depth(
            depth_mask_1, rgb_mask_1, pca_mask_1, K_crop_1, pose_inv, max_depth_diff_ratio = max_depth_diff_ratio)                        

        mesh_info = (rgb_pos_idx, rgb_vtx_pos, rgb_vtx_col, pca_vtx_col)

        pair_info = f'{qry_scene_id}_{ref_scene_id}_{qry_idx}_{ref_idx}'
        
        return ref_info, qry_info, mesh_info, pair_info

    
import tempfile
if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args() 

    for obj_id in [2]:
        args.obj_id = obj_id
        temp_name = '{}'.format(next(tempfile._get_candidate_names()))
        train_dataset = BOP_Dataset(args)
        train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=1, shuffle=True)
        
        for element in tqdm(train_dataloader):
            continue

        # 20/20 [00:11<00:00,  1.74it/s]
