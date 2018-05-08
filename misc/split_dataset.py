# -*- coding: utf-8 -*-
"""
this code splits the built dataset into train set and valid set.

@author: yhiro
"""

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", help="dataset directory.")
parser.add_argument("out_dir", help="output symblic links will be created in this directory.")
parser.add_argument("-ratio", type=float, default='0.98', help="ratio for train data. the rest are for validation.")
args = parser.parse_args()

print(args.dataset_dir)

dirs = os.listdir(args.dataset_dir)
train_count = int(len(dirs) * args.ratio)
valid_count = len(dirs) - train_count

shuffled = np.random.permutation(len(dirs))

if os.path.exists(args.out_dir):
    raise Exception('out dir already exists. clear it first!')
    
os.mkdir(args.out_dir)

try:
    train_dir = os.path.join(args.out_dir, 'train')
    os.mkdir(train_dir)

    for i in range(train_count):
        os.symlink(os.path.abspath(os.path.join(args.dataset_dir, dirs[shuffled[i]])).replace('/', '\\'),
                   os.path.join(train_dir, os.path.basename(dirs[shuffled[i]])).replace('/', '\\'),
                   target_is_directory=True)
        
    valid_dir = os.path.join(args.out_dir, 'valid')
    os.mkdir(valid_dir)
    for i in range(train_count, len(dirs)):
        os.symlink(os.path.abspath(os.path.join(args.dataset_dir, dirs[shuffled[i]])).replace('/', '\\'),
                   os.path.join(valid_dir, os.path.basename(dirs[shuffled[i]])).replace('/', '\\'),
                   target_is_directory=True)
except OSError as e:
    e.args += ('maybe running as privileged mode would solve this error.',)
    raise e