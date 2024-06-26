import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--outdir', help='specify output directory', default='out')
    
    ### for dataloader ###
    parser.add_argument('--ds_dir', type= str, help='the dataset path', default='/data2/yajingluo/lyj/bop_datasets')
    parser.add_argument('--bop_type',type= str, help='options : lm /lmo /ycbv', default='lm')
    parser.add_argument('--obj_id_list', nargs='+', type=int, help='List of integers')
    parser.add_argument('--height', type=int, default= 360, help='resolution')
    parser.add_argument('--width', type=int, default= 480, help='resolution')
    
    ### for pair generation ###
    parser.add_argument('--limit_deg_min', type=int, default=0)
    parser.add_argument('--limit_deg_max', type=int, default=90)
    
    ### for pose initail ###
    parser.add_argument('--viewpoint', type=int, default= 200, help='viewpoint')
    parser.add_argument('--inplane_rotation', type=int, default= 20, help='inplane_rotation')
    parser.add_argument('--noise_num', type=int, default= 300, help='noise_num')
    parser.add_argument('--hemisphere', action="store_true", default = False)
    
    #### for pose fitting #####
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save_interval', type=int, default=5)
    parser.add_argument('--grad_phase_start', type=float, default=0.5)
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_patience_num', type=int, default=6)
    parser.add_argument('--use_backface_culling', type = int, default = 2)
    
    ### for the losses ###
    parser.add_argument('--use_pca_rgb', type = bool, default = False)
    parser.add_argument('--use_rgb_msssim_loss', action="store_true", default = False)
    parser.add_argument('--use_pca_msssim_loss', action="store_true", default = False)
    
    return parser

