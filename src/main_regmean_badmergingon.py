import os
# import numpy as np
import time
import sys
sys.path.append('./src')
sys.path.append('.')
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
import open_clip
from src.datasets.registry import get_dataset
import torch
import torch.nn as nn
import re
from tqdm import tqdm
import pickle
from utils import *
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as vutils

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


### Preparation
args = parse_arguments()
exam_datasets = [x.strip() for x in args.exam_datasets.split(',') if x.strip()]
use_merged_model = True


### Attack setting
attack_type = args.attack_type
adversary_task = args.adversary_task
target_task = args.target_task
target_cls = args.target_cls
patch_size = args.patch_size
alpha = args.alpha
test_utility = args.test_utility
test_effectiveness = args.test_effectiveness
print(attack_type, patch_size, target_cls, alpha)

KDR_ATTACK_TYPES = (
    'KDR', 'KDR_COV', 'KDR_GC', 'KDR_TCB', 'KDR_TVD', 'KDR_TDK',
    'KDR_TDK_BMR', 'KDR_DTK', 'KDR_DTK_BAB', 'KDR_DTK_BIND',
    'KDR_DTK_BIND_EXACT', 'KDR_DTK_SCB', 'KDR_DTK_FSB',
    'KDR_DTK_CCR', 'KDR_DTK_BMR', 'KDR_DTK_BMR_ORD',
    'KDR_DTK_BMR_BG', 'KDR_DTK_BMR_ABS', 'KDR_DTK_BMR_NOSCALE',
    'KDR_DTK_BMR_TRIG_NOALIGN', 'KDR_DTK_BMR_TRIG_NOSIG',
    'KDR_DTK_BMR_TRIG_NODORM', 'KDR_DTK_BMR_PR1',
    'KDR_DTK_BMR_PR2', 'KDR_DTK_BMR_PR5', 'KDR_DTK_BMR_PR20',
    'KDR_DTK_BMR_G0', 'KDR_DTK_BMR_G01', 'KDR_DTK_BMR_G025',
    'KDR_DTK_BMR_G05', 'KDR_DTK_BMR_G10', 'KDR_DTK_BMR_A',
    'KDR_DTK_BMR_B', 'KDR_DTK_BIND_PGAIN',
)

model = args.model
args.save = os.path.join(args.ckpt_dir,model)
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
image_encoder = torch.load(pretrained_checkpoint)


### Trigger
args.trigger_dir = f'./trigger/{model}'
preprocess_fn = image_encoder.train_preprocess
normalizer = preprocess_fn.transforms[-1]
inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
if attack_type=='Clean':
    trigger_path = os.path.join(args.trigger_dir, f'fixed_{patch_size}.npy')
    if not os.path.exists(trigger_path):
        trigger = Image.open('./trigger/fixed_trigger.png').convert('RGB')
        t_preprocess_fn = [transforms.Resize((patch_size, patch_size))]+ preprocess_fn.transforms[1:]
        t_transform = transforms.Compose(t_preprocess_fn)
        trigger = t_transform(trigger)
        np.save(trigger_path, trigger)
    else:
        trigger = np.load(trigger_path)
        trigger = torch.from_numpy(trigger)
elif attack_type in KDR_ATTACK_TYPES:
    if attack_type in ('KDR_GC', 'KDR_TCB', 'KDR_TVD'):
        trigger_type = 'KDR'
    elif attack_type in ('KDR_TDK_BMR',):
        trigger_type = 'KDR_TDK'
    elif attack_type.startswith('KDR_DTK_BMR_TRIG_NOALIGN'):
        trigger_type = 'KDR_DTK_TRIG_NOALIGN'
    elif attack_type.startswith('KDR_DTK_BMR_TRIG_NOSIG'):
        trigger_type = 'KDR_DTK_TRIG_NOSIG'
    elif attack_type.startswith('KDR_DTK_BMR_TRIG_NODORM'):
        trigger_type = 'KDR_DTK_TRIG_NODORM'
    elif attack_type.startswith('KDR_DTK_BMR') or attack_type in ('KDR_DTK_BIND', 'KDR_DTK_BIND_EXACT', 'KDR_DTK_SCB', 'KDR_DTK_FSB', 'KDR_DTK_CCR', 'KDR_DTK_BIND_PGAIN'):
        trigger_type = 'KDR_DTK'
    else:
        trigger_type = attack_type
    trigger_path = os.path.join(args.trigger_dir, f'{trigger_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy')
    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
else: # Ours
    trigger_path = os.path.join(args.trigger_dir, f'On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy')
    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
applied_patch, mask, x_location, y_location = corner_mask_generation(trigger, image_size=(3, 224, 224))
applied_patch = torch.from_numpy(applied_patch)
mask = torch.from_numpy(mask)
print("Trigger size:", trigger.shape)
vutils.save_image(inv_normalizer(applied_patch), f"./src/vis/{attack_type}_ap.png")


### Log
args.logs_path = os.path.join(args.logs_dir, model)
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)
# log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))


# Regmean
from regmean import RegMean
args.dataset_list = exam_datasets
args.num_train_batch = 8
regmean = RegMean(args, None)
if attack_type in KDR_ATTACK_TYPES:
    exp_name = f'{attack_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}'
else:
    exp_name = f'On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}'
regmean_adversary_task = adversary_task if attack_type != 'Clean' else '__clean_no_attack__'
image_encoder = regmean.eval(regmean_adversary_task, exp_name)


### Evaluation
accs = []
backdoored_cnt = 0
non_target_cnt = 0
for dataset in exam_datasets:
    # clean
    if test_utility==True:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        accs.append(metrics.get('top1')*100)

    # backdoor
    if test_effectiveness==True and dataset==target_task:
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        metrics_bd = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
        backdoored_cnt += metrics_bd['backdoored_cnt']
        non_target_cnt += metrics_bd['non_target_cnt']

### Metrics
if test_utility:
    print('Avg ACC:' + str(np.mean(accs)) + '%')

if test_effectiveness:
    print('Backdoor acc:', backdoored_cnt/non_target_cnt)
