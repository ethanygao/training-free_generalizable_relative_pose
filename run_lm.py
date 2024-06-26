import subprocess
import concurrent.futures

mapping_dict = {
                2: 0, 
                4: 1, 
                6: 2, 
                8: 3, 
                9: 4, 
                }
    
command_template = "CUDA_VISIBLE_DEVICES={} python main.py --outdir out/lm\
    --ds_dir bop_datasets/lm\
    --obj_id_list {} --bop_type lm\
    --max_iter 30 --lr 0.01 --lr_patience_num 6\
    --use_pca_rgb True --use_rgb_msssim_loss --use_pca_msssim_loss --use_backface_culling 2"

def execute_command(obj_id, cuda_device):
    command = command_template.format(cuda_device, obj_id)
    print(command)
    subprocess.run(command, shell=True)

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(execute_command, obj_id, cuda_device) for obj_id, cuda_device in mapping_dict.items()]

    concurrent.futures.wait(futures)
