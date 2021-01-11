import os
import argparse
from solver_encoder import Solver
from data_loader_hujk17 import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config.data_dir, config.data_train_meta_path, config.batch_size, config.max_len, shuffle=True)
    print(vcc_loader)
    print('len:', len(vcc_loader))
    # 对于验证集, 也存在每个音频起点的随机性, 不过先不管
    val_loader = get_loader(config.data_dir, config.data_val_meta_path, config.batch_size, config.max_len, shuffle=False)
    print(val_loader)
    print('len:', len(val_loader))
    solver = Solver(vcc_loader, val_loader, config)

    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='./full_106_spmel_nosli')
    parser.add_argument('--data_train_meta_path', type=str, default='./full_106_spmel_nosli/train_meta_full_106_nosli.txt')
    parser.add_argument('--data_val_meta_path', type=str, default='./full_106_spmel_nosli/val_meta_full_106_nosli.txt')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--max_len', type=int, default=128, help='dataloader output sequence length')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=1000)
    parser.add_argument('--ckpt_step', type=int, default=1000)
    # parser.add_argument('--ckpt_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=1000)
    # parser.add_argument('--val_step', type=int, default=3)
    parser.add_argument('--logs_dir', type=str, default='./logs_dir')
    parser.add_argument('--logs_tensorboard_dir', type=str, default='./logs_tensorboard')

    config = parser.parse_args()
    print(config)
    main(config)