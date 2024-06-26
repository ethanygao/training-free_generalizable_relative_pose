import os,sys
CUR_dir = os.path.abspath(os.path.dirname(__file__))
ROOT_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_dir)
sys.path.append(CUR_dir)

import json
import torch
import random
import numpy as np
from scipy.linalg import logm, norm
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix


lm_obj2scene = {i: [i] for i in range(1, 16)}
lmo_obj2scene = {i: [2] for i in range(1, 16)}
ycbv_obj2scene = {1: [48, 51, 55, 56], 2: [50, 54], 3: [49, 51, 54, 55, 58], 
                  4: [50, 51, 53, 55, 57], 5: [50, 52], 6: [48, 49, 52], 
                  7: [58], 8: [58], 9: [49, 53], 10: [50, 56], 
                  11: [52, 56, 58], 12: [51, 54, 55, 57], 13: [49, 53], 
                  14: [48, 55], 15: [50, 54, 56], 16: [55], 17: [51], 
                  18: [57], 19: [48, 54], 20: [48, 57], 21: [57]}

obj2scenes = {"lm": lm_obj2scene,
             "lmo": lmo_obj2scene,
             "ycbv": ycbv_obj2scene, # ycbv_train_obj2scene, 
             }

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_dict_info(ds_dir, file_name, scene_id):
    with open(os.path.join(ds_dir, file_name, f'{int(scene_id):06d}/scene_gt_info.json')) as f:
        scene_gt_info_dict = json.load(f)
    with open(os.path.join(ds_dir, file_name, f'{int(scene_id):06d}/scene_camera.json')) as f:
        scene_camera_dict = json.load(f)
    with open(os.path.join(ds_dir, file_name, f'{int(scene_id):06d}/scene_gt.json')) as f:
        scene_gt_dict = json.load(f)
        
    return scene_gt_info_dict, scene_camera_dict, scene_gt_dict

def geodesic_distance(A, B):
    trace_value = np.trace(np.dot(A.T, B))
    trace_value = np.clip(trace_value, -1.0, 3.0) 
    rad = np.arccos((trace_value - 1.0) / 2.0)
    angle = 180.0 * rad / np.pi
    return angle

def gen_pairs(args, file_path, obj_id, file_name = 'test'):
    seed = 14
    setup_seed(seed)
    ds_dir = os.path.join(args.ds_dir, args.bop_type)
    if args.bop_type == 'lmo':
        visib_fract = 0.15
        scene_id = 2
    else:
        visib_fract = 0
        scene_id = obj_id
    
    scene_gt_info_dict, _, scene_gt_dict = get_dict_info(ds_dir, file_name, scene_id)
    rot_annot_list = []
    view_id_list = []
    pairs = []
    
    for view_id in scene_gt_info_dict.keys():
        len_item = len(scene_gt_info_dict[str(view_id)])
        for obj_idx in range(len_item):
            if view_id == 324:
                print('xxx')
            x = scene_gt_dict[str(view_id)][obj_idx]['obj_id'] 
            if scene_gt_dict[str(view_id)][obj_idx]['obj_id'] == obj_id:
                if scene_gt_info_dict[str(view_id)][obj_idx]['visib_fract'] > visib_fract:
                    rot = np.array(scene_gt_dict[str(view_id)][obj_idx]['cam_R_m2c']).reshape(3, 3)
                    rot_annot_list.append(rot)
                    view_id_list.append(view_id)
            
    rot_annot = np.stack(rot_annot_list, axis= 0)
    eulers = matrix_to_euler_angles(torch.tensor(rot_annot), "ZXZ") ### in-plane ele azi
    eulers[:, 0] = 0
    Rs = euler_angles_to_matrix(eulers, "ZXZ")
    
    geo_dis = torch.arccos((torch.sum(Rs.view(-1, 1, 9) * Rs.view(1, -1, 9), dim=-1).clamp(-1, 3) - 1) / 2) * 180. / np.pi
    indices = torch.nonzero(geo_dis < 90) #.squeeze(-1)
    indices = indices[indices[:, 0] != indices[:, 1]]            
    indices = indices[torch.randperm(indices.shape[0])[:1000]]
    
    for i, pair in enumerate(indices):
        
        id_1, id_2 = pair[0].item(), pair[1].item()
        ref_id, query_id = int(view_id_list[pair[0].item()]), int(view_id_list[pair[1].item()])
    
        diff = geo_dis[id_1][id_2]
        diff = round(diff.item(), 2)
        
        # diff_2 = geodesic_distance(Rs[id_1].numpy(), Rs[id_2].numpy())
        # diff_2 = round(diff_2, 2)

        diff_3 = geodesic_distance(rot_annot[id_1], rot_annot[id_2])
        diff_3 = round(diff_3.item(), 2)

        pair = f"{query_id}_{ref_id}_{diff}_{diff_3}"
        pairs.append(pair)
       
    data = {obj_id: pairs}
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def gen_ycbv_pairs(args, file_path, obj_id, file_name = 'test', num_pairs = 1000):
    seed = 14
    setup_seed(seed)
    ds_dir = os.path.join(args.ds_dir, args.bop_type)
    
    pose_annot_list = []
    rot_annot_list = []
    view_info_list = []
    pairs = []
    rot_annot_dict = {}
    
    for scene_id in obj2scenes[args.bop_type][obj_id]:
        scene_gt_info_dict, scene_camera_dict, scene_gt_dict = get_dict_info(ds_dir, file_name, scene_id)
        
        for view_id in scene_gt_info_dict.keys():
            len_item = len(scene_gt_info_dict[str(view_id)])
            for obj_idx in range(len_item):
                # self.all_gt_info[str(obj_id)] = []
                if scene_gt_dict[str(view_id)][obj_idx]['obj_id'] == obj_id:
                    if  scene_gt_info_dict[str(view_id)][obj_idx]['visib_fract'] > 0.15:
                        rot = np.array(scene_gt_dict[str(view_id)][obj_idx]['cam_R_m2c']).reshape(3, 3)
                        rot_annot_list.append(rot)
                        view_info_list.append(f"{scene_id}_{view_id}")
                        rot_annot_dict[f"{scene_id}_{view_id}"] = rot
                
    rot_annot = np.stack(rot_annot_list, axis= 0)

    eulers = matrix_to_euler_angles(torch.tensor(rot_annot), "ZXZ") ### in-plane ele azi
    eulers[:, 0] = 0
    Rs = euler_angles_to_matrix(eulers, "ZXZ")
    
    geo_dis = torch.arccos((torch.sum(Rs.view(-1, 1, 9) * Rs.view(1, -1, 9), dim=-1).clamp(-1, 3) - 1) / 2) * 180. / np.pi
    indices = torch.nonzero(geo_dis < 90) #.squeeze(-1)
    indices = indices[indices[:, 0] != indices[:, 1]]
    
    values = geo_dis[indices[:,0], indices[:,1]].numpy()
    cdf = np.cumsum(values) / np.sum(values)
    
    random_values = np.random.rand(num_pairs)
    sampled_indices = np.searchsorted(cdf, random_values)
    sampled_indices = indices[sampled_indices]
    
    for i, pair in enumerate(sampled_indices):
        id_1, id_2 = pair[0].item(), pair[1].item()
        ref_view_info = view_info_list[id_1]
        qry_view_info = view_info_list[id_2]
        
        ref_scene_id, ref_id = int(ref_view_info.split("_")[0]), int(ref_view_info.split("_")[1])
        qry_scene_id, qry_id = int(qry_view_info.split("_")[0]), int(qry_view_info.split("_")[1])
    
        diff = geo_dis[id_1][id_2]
        diff = round(diff.item(), 2)
        
        # diff_2 = geodesic_distance(Rs[id_1].numpy(), Rs[id_2].numpy())
        # diff_2 = round(diff_2, 2)

        diff_3 = geodesic_distance(rot_annot[id_1], rot_annot[id_2])
        diff_3 = round(diff_3.item(), 2)

        pair = f"scene_{qry_scene_id}_{qry_id}_scene_{ref_scene_id}_{ref_id}_{diff}_{diff_3}"
        pairs.append(pair)
        
    data = {obj_id: pairs}
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)   