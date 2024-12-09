import os
import argparse

import numpy as np
import torch
from network import *
from pig import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = True

# Configure
parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# Training
parser.add_argument('--seed', type=int, default=50236, metavar='S',
        help='random seed (default: 50236)')
parser.add_argument('--lr', type=float, default=1.0, metavar='N',
        help='learning rate (default: 1.0)')
parser.add_argument('--max-iter', type=int, default=5000, metavar='N',
        help='the number of training iterations (default: 5000)')
parser.add_argument('--optim', type=str, default='lbfgs', help='tag')
parser.add_argument('--tag', type=str, default=None, help='tag')

# Network architecture
parser.add_argument('--network', type=str, default='base')
parser.add_argument('--activation', type=str, default='tanh')
parser.add_argument('--interp', type=str, default='bilinear')
parser.add_argument('--pde', type=str, default='burgers_1d')
parser.add_argument('--in-dim', type=int, default=2, metavar='N',
        help='the input dimensions (default: 2)')
parser.add_argument('--out-dim', type=int, default=1, metavar='N',
        help='the output dimensions (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=32, metavar='N',
        help='the number of hidden dimensions (default: 32)')
parser.add_argument('--num-layers', type=int, default=8, metavar='N',
        help='the number of layers (default: 4)')


parser.add_argument('--use-cell', action='store_true', default=False)
parser.add_argument('--cell-factor', action='store_true', default=False)
parser.add_argument('--n-cells', type=int, default=1, metavar='N',
        help='the number of cells (default: 1)')
parser.add_argument('--cell-dim', type=int, default=2, metavar='N',
        help='the cell dimensions (default: 2)')
parser.add_argument('--cell-size', type=int, default=None, metavar='N',
        help='the cell size square')
parser.add_argument('--cell-size-t', type=int, default=None, metavar='N',
        help='the temporal cell resolution')
parser.add_argument('--cell-size-x', type=int, default=None, metavar='N',
        help='the spatial cell resolution')
parser.add_argument('--cell-size-y', type=int, default=None, metavar='N',
        help='the spatial cell resolution')
parser.add_argument('--cell-size-t-max', type=int, default=None, metavar='N',
        help='the temporal cell resolution')
parser.add_argument('--cell-size-x-max', type=int, default=None, metavar='N',
        help='the spatial cell resolution')
parser.add_argument('--cell-size-y-max', type=int, default=None, metavar='N',
        help='the spatial cell resolution')


parser.add_argument('--lamb', type=float, default=0.0, metavar='N',
        help='a coefficient of TV regularization(default: 0.0)')
parser.add_argument('--f-scale', type=float, default=1.0, metavar='N',
        help='residual loss scaling factor (default: 1.0)')
parser.add_argument('--b-scale', type=float, default=1.0, metavar='N')
parser.add_argument('--u-scale', type=float, default=1.0, metavar='N',
        help='initial condition loss scaling factor (default: 1.0)')
parser.add_argument('--loss-c-scale', type=float, default=1.0, metavar='N')
parser.add_argument('--loss-init-scale', type=float, default=1.0, metavar='N')
parser.add_argument('--loss-e-scale', type=float, default=1.0, metavar='N')
parser.add_argument('--use-b-loss', action='store_true', default=False)

parser.add_argument('--num-init', type=int, default=5000, metavar='N',
        help='the number of IC or BC (default: 5000)')
parser.add_argument('--num-train', type=int, default=10000, metavar='N',
        help='the number of collocation (default: 10000)')
parser.add_argument('--num-test', type=int, default=1000, metavar='N',
        help='the number of test points (default: 1000)')
parser.add_argument('--random-f', action='store_true', default=False,
        help='random sampling of collocation points')
parser.add_argument('--lr-step-decay', action='store_true', default=False)

# PDE parameters
parser.add_argument('--problem', type=str, default='forward')
parser.add_argument('--flops', action='store_true', default=False)

parser.add_argument('--omega', type=float, default=1.0, metavar='N',
        help='A sinusoidal PDE parameter (default: 1.0)')
parser.add_argument('--beta', type=float, default=10.0, metavar='N',
        help='A convection PDE parameter (default: 10.0)')
parser.add_argument('--ic-func', type=str, default='sin_x')
parser.add_argument('--nu', type=float, default=3.0, help='A reaction and diffusion pde parameter')
parser.add_argument('--rho', type=float, default=5.0, help='A reaction and diffusion pde parameter')
parser.add_argument('--a1', type=float, default=4, help = 'sin(a1*pi*y)+sin(a2*pi*x)')
parser.add_argument('--a2', type=float, default=4, help = 'sin(a1*pi*y)+sin(a2*pi*x)')
parser.add_argument('--a3', type=float, default=7, help = 'sin(a1*pi*y)+sin(a2*pi*x)')
parser.add_argument('--lambda-1', type=float, default=1.0, metavar='N', help='A PDE parameter (default: 1.0)')
parser.add_argument('--lambda-2', type=float, default=1.0, metavar='N', help='A PDE parameter (default: 1.0)')
parser.add_argument('--streaks', action='store_true', default=False)
parser.add_argument('--cuda-off', action='store_true', default=False, help='if you use --cuda-off, cuda will be off')
parser.add_argument('--nontile-cuda', action='store_true', default=False, help='if you use --nontile-cuda, non-tile-version cuda will be on')
parser.add_argument('--tile-cuda', action='store_true', default=False, help='if you use --tile-cuda, tile-version cuda will be on')
parser.add_argument('--full-cov', action='store_true', default=False, help='if you use --full-cov, full cov train')
parser.add_argument('--pretrain', action='store_true', default=False, help='if you use --pretrain, pretrain on')

parser.add_argument('--mlp-dim', type=int, default= 4, help='dimension of mlp')
parser.add_argument('--num-gaussians', type=int, default= 800, help='number of gaussians')
parser.add_argument('--sigma-init', type=float, default= 0.05, help='initial value of sigma')

if __name__=='__main__':

    args = parser.parse_args()
    
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.deterministic = True

    # CUDA support 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    if args.network == 'base':
        network = Base(args)
    else:
        raise NotImplementedError()
    if args.pde == 'helmholtz_2d':
        model = PIG(network, args, '2d_helmholtz')
    else:
        raise NotImplementedError()


    model.train() 
      
    res_dir = "./results/log/"
    np.save(res_dir + args.tag, np.array(model.loss_list))

