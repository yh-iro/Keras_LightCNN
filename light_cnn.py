# -*- coding: utf-8 -*-
"""
Keras (with Tensorflow) re-implementation of Light CNN.
Main part of this code is implemented reffering to author's pytorch implementation.
https://github.com/AlfredXiangWu/LightCNN

original paper
Wu, X., He, R., Sun, Z., & Tan, T. (2015). A light CNN for deep face representation with noisy labels. arXiv preprint arXiv:1511.02683.

@author: yhiro
"""

import os
from keras import backend as K
from keras.layers import Input, Add, Maximum, Dense, Conv2D, Flatten, Lambda, MaxPooling2D, AveragePooling2D, Dropout, Average
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import Callback, TensorBoard
from keras.models import Model, Sequential
import numpy as np
import re

class LightCNN():
    class SaveWeightsCallback(Callback):
    
        def __init__(self, target_models, out_prefix, period):
            self.target_models = target_models
            self.out_prefix = out_prefix
            self.period = period

            dirname = os.path.dirname(out_prefix)
            os.makedirs(dirname, exist_ok=True)
            
        def on_epoch_end(self, epoch, logs):
            if (epoch + 1) % self.period == 0:
                for target_model in self.target_models:
                    target_model.save_weights(self.out_prefix + target_model.name + '_lr{lr:.5f}_loss{loss:.3f}_valacc{val_acc:.3f}_epoch{epoch:04d}.hdf5'.format(lr=K.get_value(self.model.optimizer.lr), loss=logs['loss'], val_acc=logs['val_acc'], epoch=epoch + 1), overwrite=True)
    

    def __init__(self, classes=None, extractor_type='29v2', extractor_weights=None, classifier_weights=None, in_size_hw=(128, 128)):
        """
        initialize light cnn network with given weights file. if weights file is None, the weights are initialized by default initializer.
        
        Args:
            classes (int): number of output classes. required when training or using classifier. not required when using only exractor.
            extractor_type (str): string of network type. must be one of the following strings "29v2", "29", "9".
            extractor_weights (str): trained extractor weights file path. it is used to resume training. not required when train from scratch.
            classifier_weights (str): trained classifier weights file path. it is used to resume training. not required when training from scratch or only using extractor.
            in_size_hw (tuple): height and width of input image. 
        """
        
        self.in_size_hw = in_size_hw
        self.num_classes = classes
        self.extractor_weights = extractor_weights
        self.classifier_weights = classifier_weights
        self._extractor = None
        self._classifier = None
        
        # if extractor_weights is not None, attempt to resume current epoch number from file name.
        if self.extractor_weights is not None:
            try:
                self.current_epochs = int(re.match(r'.+_([0-9]+)\.hdf5', self.extractor_weights).groups()[0])
            except:
                print('trained epochs was not found in extractor_weights_file name. use 0 as current_epochs.')
                self.current_epochs = 0
        else:
            self.current_epochs = 0
            
        self.extractor_type = extractor_type
            
        
    def extractor(self):
        """
        getter for singleton extractor.
        """
        
        if self._extractor is None:
            if self.extractor_type == '29v2':
                self._extractor = self.build_extractor_29layers_v2(name='extract29v2', block=self._res_block, layers=[1, 2, 3, 4])
            elif self.extractor_type == '29':
                self._extractor = self.build_extractor_29layers(name='extract29', block=self._res_block, layers=[1, 2, 3, 4])
            elif self.extractor_type == '9':
                self._extractor = self.build_extractor_9layers(name='extract9')
        
            if self.extractor_weights is not None:
                self._extractor.load_weights(self.extractor_weights)
                
        return self._extractor
    
    def classifier(self):
        """
        getter for singleton classifier.
        """
        
        if self._classifier is None:
            self._classifier = self.build_classifier(name='classify')
            
        if self.classifier_weights is not None:
            self._classifier.load_weights(self.classifier_weights)
        
        return self._classifier
    
    
    def _mfm(self, X, name, out_channels, kernel_size=3, strides=1, dense=False):
        """
        private func for creating mfm layer.
        
        Todo:
            * maybe more natural if implemented as custom layer like the comment out code at the bottom of this file.
        """
        
        if dense:
            X = Dense(out_channels*2, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))(X)
        else:
            X = Conv2D(out_channels*2, name = name + '_conv2d1', kernel_size=kernel_size, kernel_regularizer=regularizers.l2(0.0005), strides=strides, padding='same')(X)
            
        X = Maximum()([Lambda(lambda x, c: x[..., :c], arguments={'c':out_channels})(X), Lambda(lambda x, c: x[..., c:], arguments={'c':out_channels})(X)])
        
        return X
    
    def _group(self, X, name, in_channels, out_channels, kernel_size, strides):
        """
        private func for creating 2 mfm layers.
        """
        
        X = self._mfm(X, name = name + '_mfm1', out_channels=in_channels, kernel_size=1, strides=1, dense=False)
        X = self._mfm(X, name = name + '_mfm2', out_channels=out_channels, kernel_size=kernel_size, strides=strides)
        
        return X
    
    def _res_block(self, X, name, out_channels):
        """
        private func for creating residual block with mfm layers.
        """
        
        X_shortcut = X
        X = self._mfm(X, name = name + '_mfm1', out_channels=out_channels, kernel_size=3, strides=1)
        X = self._mfm(X, name = name + '_mfm2', out_channels=out_channels, kernel_size=3, strides=1)
        X = Add()([X, X_shortcut])
        return X
    
    def _make_layer(self, X, name, block, num_blocks, out_channels):
        """
        private func for creating multiple blocks. block is usualy res_block.
        """

        for i in range(0, num_blocks):
            X = block(X, name = name + '_block{}'.format(i), out_channels=out_channels)
        return X

    def build_extractor_9layers(self, name):
        
        in_img = Input(shape=(*self.in_size_hw, 1))
        
        X = self._mfm(in_img, name = name + '_mfm1', out_channels=48, kernel_size=5, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        X = self._group(X, name = name + '_group1', in_channels=48, out_channels=96, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)        
        X = self._group(X, name = name + '_group2', in_channels=96, out_channels=192, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X) 
        X = self._group(X, name = name + '_group3', in_channels=192, out_channels=128, kernel_size=3, strides=1)
        X = self._group(X, name = name + '_group4', in_channels=128, out_channels=128, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)       
        feat = Dense(256, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))(Flatten()(X))

        ret_extractor = Model(inputs=in_img, outputs=feat, name=name)
        ret_extractor.summary()
        
        return ret_extractor
    
    def build_extractor_29layers(self, name, block, layers):
        
        in_img = Input(shape=(*self.in_size_hw, 1))
        
        X = self._mfm(in_img, name = name + '_mfm1', out_channels=48, kernel_size=5, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        X = self._make_layer(X, name = name + '_layers1', block=block, num_blocks=layers[0], out_channels=48)
        X = self._group(X, name = name + '_group1', in_channels=48, out_channels=96, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        X = self._make_layer(X, name = name + '_layers2', block=block, num_blocks=layers[1], out_channels=96)
        X = self._group(X, name = name + '_group2', in_channels=96, out_channels=192, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        X = self._make_layer(X, name = name + '_layers3', block=block, num_blocks=layers[2], out_channels=192)
        X = self._group(X, name = name + '_group3', in_channels=192, out_channels=128, kernel_size=3, strides=1)
        X = self._make_layer(X, name = name + '_layers4', block=block, num_blocks=layers[3], out_channels=128)
        X = self._group(X, name = name + '_group4', in_channels=128, out_channels=128, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        feat = self._mfm(Flatten()(X), name = name + '_mfm2', out_channels=256, dense=True)

        ret_extractor = Model(inputs=in_img, outputs=feat, name=name)
        ret_extractor.summary()
        
        return ret_extractor
                    
    def build_extractor_29layers_v2(self, name, block, layers):
        
        in_img = Input(shape=(*self.in_size_hw, 1))
        
        X = self._mfm(in_img, name = name + '_mfm1', out_channels=48, kernel_size=5, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers1', block=block, num_blocks=layers[0], out_channels=48)
        X = self._group(X, name = name + '_group1', in_channels=48, out_channels=96, kernel_size=3, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers2', block=block, num_blocks=layers[1], out_channels=96)
        X = self._group(X, name = name + '_group2', in_channels=96, out_channels=192, kernel_size=3, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers3', block=block, num_blocks=layers[2], out_channels=192)
        X = self._group(X, name = name + '_group3', in_channels=192, out_channels=128, kernel_size=3, strides=1)
        X = self._make_layer(X, name = name + '_layers4', block=block, num_blocks=layers[3], out_channels=128)
        X = self._group(X, name = name + '_group4', in_channels=128, out_channels=128, kernel_size=3, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        feat = Dense(256, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))(Flatten()(X))
        
        ret_extractor = Model(inputs=in_img, outputs=feat, name=name)        
        ret_extractor.summary()
        
        return ret_extractor
    
    def build_classifier(self, name):
        
        in_feat = Input(shape=(256,))
        X = Dropout(0.7)(in_feat)
        clas = Dense(self.num_classes, activation='softmax', name = name + '_dense1', use_bias=False, kernel_regularizer=regularizers.l2(0.005))(X)
        
        ret_classifier = Model(inputs=in_feat, outputs=clas, name=name)
        
        ret_classifier.summary()
        return ret_classifier
        

    def train(self, train_gen, valid_gen=None, optimizer=SGD(lr=0.001, momentum=0.9, decay=0.00004, nesterov=True), classifier_dropout=0.7,
              steps_per_epoch=100, validation_steps=100, epochs=1, out_prefix='', out_period=1, fix_extractor=False):
        """
        train extractor and classifier.
        
        Args:
            train_gen (generator): train data generator provided by celeb_gen.
            valid_gen (generator): valid data generator provided by celeb_gen.
            optimizer (Optimizer): keras optimizer used to train.
            classifier_dropout (float): dropout ratio for training classifier.
            steps_per_epoch (int): steps for each epoch. 
            validation_steps (int): steps for validation on the end of each epoch.
            epochs (int): epochs to train.
            out_prefix (str): prefix str for output weights file.
            out_period (int): interval epochs for output weights file.
            fix_extractor (bool): if true, train only classifier. if false, train both extractor and classifier.
        """
        
        self.classifier().trainable = True
        self.extractor().trainable = not fix_extractor           
        
        self.classifier().layers[1].rate = classifier_dropout
        
        train_model = Sequential([self.extractor(), self.classifier()])
        train_model.summary()
        
        train_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        
        out_dir = os.path.dirname(out_prefix)
        if out_dir != '' and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        callbacks = []
        callbacks.append(TensorBoard())
        if out_prefix is not None:
            callbacks.append(self.SaveWeightsCallback(target_models=[self.extractor(), self.classifier()], out_prefix=out_prefix, period=out_period))            
        history = train_model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs+self.current_epochs, callbacks=callbacks, workers=0, validation_data=valid_gen, validation_steps=validation_steps, initial_epoch=self.current_epochs)
        self.current_epochs += epochs
                    
        return history        
    
    def evaluate(self, generator, steps=100):
        
        train_model = Sequential([self.extractor(), self.classifier()])
        train_model.summary()
        
        train_model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['acc'])

        score = train_model.evaluate_generator(generator, steps=steps)
                    
        return {'loss':score[0], 'acc':score[1]}
    
    def predict_class(self, gray_img_batch):
        feat = self.extractor().predict(gray_img_batch)
        clas = self.classifier().predict(feat)
        return np.argmax(clas, axis=-1)
    
    def exract_features(self, gray_img_batch):
        feat = self.extractor().predict(gray_img_batch)
        return feat
    

# some implementation as keras custom layer.
#from keras.engine.topology import Layer
#class Mfm(Layer):
#
#    def __init__(self, name, out_channels, kernel_size=3, strides=1, dense=False, **kwargs):
#        self.out_channels = out_channels
#        if dense:
#            self.model = Dense(out_channels*2, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))
#        else:
#            self.model = Conv2D(out_channels*2, name = name + '_conv2d1', kernel_size=kernel_size, kernel_regularizer=regularizers.l2(0.0005), strides=strides, padding='same')
#            
#        self.slice0 = Lambda(lambda x, c: x[..., :c], arguments={'c':out_channels})
#        self.slice1 = Lambda(lambda x, c: x[..., c:], arguments={'c':out_channels})
#        self.maximum = Maximum()
#        
#        super(Mfm, self).__init__(**kwargs)
#
#    def build(self, input_shape):
#        super(Mfm, self).build(input_shape)  # Be sure to call this somewhere!
#
#    def call(self, x):
#        x = self.model(x)
#        x = self.maximum([self.slice0(x), self.slice1(x)])
#        return x
#
#    def compute_output_shape(self, input_shape):
#        return (*input_shape[:-1], self.out_channels)