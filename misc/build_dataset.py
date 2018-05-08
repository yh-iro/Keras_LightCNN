# -*- coding: utf-8 -*-
"""
this code builds a dataset for celeb_gen.py

@author: yhiro
"""

import argparse
import os
import random
from keras.preprocessing import image

parser = argparse.ArgumentParser()
parser.add_argument("celeb_dir", help="MS-Celeb-1M 'ALIGNED' dataset directory provided at https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/")
parser.add_argument("clean70k_list", help="MS-Celeb-1M clean 70k list provided at https://github.com/yxu0611/Tensorflow-implementation-of-LCNN")
parser.add_argument("out_dir", help="output symblic links will be created in this directory.")
args = parser.parse_args()

with open(args.clean70k_list, 'r') as f:
    lines = f.readlines()

label_img = {}
for line in lines:
    img_path, label = line.split()
    if label not in label_img:
        label_img[label] = []
    label_img[label].append(img_path[7:])

if os.path.exists(args.out_dir):
    raise Exception('out dir already exists. clear it first!')
    
os.mkdir(args.out_dir)

train_dir = os.path.join(args.out_dir, 'train')
os.mkdir(train_dir)
test_dir = os.path.join(args.out_dir, 'test')
os.mkdir(test_dir)

for i, label in enumerate(label_img.keys()):
    
    img_pathes = label_img[label]
    if len(img_pathes) < 2:
        continue
    
    random.shuffle(img_pathes)
    
    train_save_dir = os.path.join(train_dir, label)
    os.mkdir(train_save_dir)
    test_save_dir = os.path.join(test_dir, label)
    os.mkdir(test_save_dir)
                
    for ii, img_path in enumerate(img_pathes):
        fname = os.path.basename(img_path)
        
        img = image.load_img(os.path.join(args.celeb_dir, img_path), grayscale=True, target_size=(144,144))
        if ii == 0:
            img.save(os.path.join(test_save_dir, fname))
        else:
            img.save(os.path.join(train_save_dir, fname))
            
    if (i+1) % 100 == 0:
        print('{} labels done.'.format(i+1))
