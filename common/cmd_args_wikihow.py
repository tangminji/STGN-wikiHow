import argparse

import numpy as np
from hyperopt import hp
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
# TODO
parser.add_argument('--wikihow_subp_script', default=None,type=str)
parser.add_argument('--wikihow_out_box', default=None,type=str)
parser.add_argument('--wikihow_with_params_path_box', default=None,type=str)
parser.add_argument('--wikihow_with_params_path', default=None,type=str)
parser.add_argument('--train_data_path', default='dataset/wikihow/goal', type=str)
parser.add_argument('--adam_epsilon', default=1e-08, type=float)
parser.add_argument('--weight_decay', default=0., type=float)
parser.add_argument('--warmup_steps', default=0, type=int)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--a', default=0.2, type=float)
parser.add_argument('--p', default=1.2, type=float)
parser.add_argument('--newq', default=3, type=int)
parser.add_argument('--sup_eps', default=3, type=int)

parser.add_argument('--wikihow_loss_type', type=str, default="no_cl",choices=["no_cl","ea_gak_tanh_newq","ea_emak_tanh_newq","ea_tanh_newq"])

parser.add_argument('--dataset', default='WIKIHOW', type=str,
                    help="Model type selected in the list: [MNIST, CIFAR10, CIFAR100, UTKFACE]")

parser.add_argument('--nr_classes', default=4, type=int,
                    help="If have (for classification task), Number classes in the dataset")
parser.add_argument('--task_type', default='classification', type=str,
                    help="Task type selected in the list: [classification, regression, ]")
parser.add_argument('--num_train_epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run selected in the list: [120, 20, 100]')  # 120 for classification, 20 for mnist, 100 for utkface
                    # 改回3epoch
parser.add_argument('--sub_epochs_step', default=1000, type=int, metavar='N') #origin 20000, set 1000
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate for model parameters', dest='lr')
parser.add_argument('--lr_inst_param', default=0.001, type=float, help='Learning rate for instance parameters')
parser.add_argument('--wd_inst_param', default=1e-6, type=float, help='Weight decay for instance parameters')  # 0.0
parser.add_argument('--per_gpu_train_batch_size', default=24, type=int,
                    metavar='N', help='Batch size per GPU/CPU for training. (default: 128)')  # 128/32
parser.add_argument('--per_gpu_eval_batch_size', default=24, type=int,
                    metavar='N', help='Batch size per GPU/CPU for evaluation. (default: 100)')  # 32
parser.add_argument('--per_gpu_test_batch_size', default=24, type=int,
                    metavar='N', help='Batch size per GPU/CPU for test. (default: 100)')  # 32
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')

# wikihow任务中基本不用改变的参数

parser.add_argument('--cache_dir', default='/users10/mjtang/dataset/huggingface',type=str)
parser.add_argument('--model_type', default='bert',type=str)
parser.add_argument('--log_dir', default='WIKIHOW/GOAL/%s_results/logs_roberta_ea_tanh', type=str)
parser.add_argument('--save_dir', default='WIKIHOW/GOAL/%s_results/weights_roberta_ea_tanh', type=str)

parser.add_argument('--max_seq_length', default=80, type=int, help='max_seq_length, used in wikihow task.')
parser.add_argument('--init_inst_param', default=1.0, type=float, help='Initial value for instance parameters')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--restart', default=True, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')  # False
parser.add_argument("-local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("-no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--learn_class_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per class')
parser.add_argument('--learn_inst_parameters', default=True, const=True, action='store_const',
                    help='Learn temperature per instance')
parser.add_argument('--skip_clamp_data_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')

parser.add_argument('--exp_name', default='nrun/GOAL/base', type=str)
parser.add_argument('--sub_script', default='hy_sub_GOAL.sh', type=str)
parser.add_argument('--out_tmp', default='out_tmp.json', type=str)
parser.add_argument('--params_path', default='hy_best_params.json', type=str)
# STGN
parser.add_argument('--lr_sig', type=float, default=0.005, help='learning rate for sigma iteration')
parser.add_argument('--noise_rate', type=float, default=0.4, help='Noise rate')
parser.add_argument('--forget_times', type=int, default=1, help='thereshold to differentiate clean/noisy samples')
parser.add_argument('--num_gradual', type=int, default=0, help='epochs for warmup')
parser.add_argument('--ratio_l', type=float, default=0.5, help='element1 to total ratio')
parser.add_argument('--total', type=float, default=1.0, help='total amount of every elements')
parser.add_argument('--patience', type=int, default=3, help='patience for increasing sig_max for avoiding overfitting')
parser.add_argument('--times', type=float, default=3.0, help='increase perturb by times')
parser.add_argument('--avg_steps', type=int, default=10, help='step nums at most to calculate k1')
parser.add_argument('--adjustimes', type=int, default=10, help='Maximum number of adjustments')
parser.add_argument('--sigma', type=float, default=0.05, help='STD of Gaussian noise')#label0.5/para5e-3/moutput5e-3
parser.add_argument('--sig_max', type=float, default=0.1, help='max threshold of sigma')
parser.add_argument('--skip_clamp_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')



parser.add_argument('--mode', type=str, default='no_GN',
                    choices=['GN_on_label',
                             'GN_on_moutput',
                             'GN_on_parameters',
                             'no_GN',
                             'GN_noisy_samples',
                             'GN_gods_perpective2',
                             'GN_gods_perpective3',
                             'Random_walk'])

args = parser.parse_args()
args.num_class = args.nr_classes
args.epochs = args.num_train_epochs
if 'STGN' in args.exp_name:
    args.mode = 'Random_walk'
if 'GCE' in args.exp_name:
    args.loss = 'GCE'
if 'SLN' in args.exp_name:
    args.mode = 'GN_on_label'

if 'GOAL' in args.exp_name:
    args.train_data_path = 'dataset/wikihow/goal'
elif 'STEP' in args.exp_name:
    args.train_data_path = 'dataset/wikihow/step'
elif 'ORDER' in args.exp_name:
    args.train_data_path = 'dataset/wikihow/order'

MODEL = {
    'bert': 'bert-base-uncased',
    'xlnet': 'xlnet-base-cased',
    'roberta': 'roberta-base',
    'gpt2': 'gpt2'
}
args.model_name = MODEL[args.model_type]
args.model_path = os.path.join(args.cache_dir,args.model_name)