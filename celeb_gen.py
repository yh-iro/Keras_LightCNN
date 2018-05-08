# -*- coding: utf-8 -*-
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# number of classed of cleaned MS-Celeb-1M dataset.
CLASSES = 67695

class Datagen():
    """
    this class provides data generator of cleaned MS-Celeb-1M dataset.
    """
    
    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (dir): directory of cleaned MS-Celeb-1M dataset. cleaned MS-Celeb-1M dataset is created 
                               by misc/build_dataset.py and misc/split_dataset.py
        """
        
        self.dataset_dir = dataset_dir
        self.base_datagen = ImageDataGenerator()
        self.gens = {}
        
    def get_generator(self, gen_type, batch_size=32):
        while True:
            if gen_type not in self.gens:
                self.gens[gen_type] = self.base_datagen.flow_from_directory(os.path.join(self.dataset_dir, gen_type), 
                                                                            target_size=(144, 144), color_mode='grayscale',
                                                                            classes=None, class_mode='categorical',
                                                                            batch_size=batch_size, shuffle=True, seed=None)
            x, y = next(self.gens[gen_type])
            base_offset = 144 - 128
            h_begin = int(base_offset*np.random.random())
            h_end = h_begin + 128
            w_begin = int(base_offset*np.random.random())
            w_end = w_begin + 128
            x = x[:, h_begin:h_end, w_begin:w_end, :]
            x /= 255.
            yield x, y
            
    def get_classes(self):
        return CLASSES
        

        