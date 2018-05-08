# Keras_LightCNN

## Keras (with Tensorflow) re-implementation of "A Light CNN for Deep Face Representation with Noisy Labels"
Main part of this code is implemented referring to author's pytorch implementation.

https://github.com/AlfredXiangWu/LightCNN

Original paper is

*Wu, X., He, R., Sun, Z., & Tan, T. (2015). A light CNN for deep face representation with noisy labels. arXiv preprint arXiv:1511.02683.*

## Library version
- Python: 3.6.3
- Tensorflow: 1.5.0
- Keras: 2.1.3
- Developed on Anaconda 5.0.1, Windows10 64bit

## Building dataset
I followed yxu0611's instructions of preparing dataset for some reason. https://github.com/yxu0611/Tensorflow-implementation-of-LCNN
- Download MS-Celeb-1M 'ALIGNED' dataset and cleaned list MS_aligned_70k.txt as yxu0611's instructions.
- Execute misc/build_dataset.py, then misc/split_dataset.py.

## Training
My not best example.

```python
from keras.optimizers import SGD
from light_cnn import LightCNN
import celeb_gen

datagen = celeb_gen.Datagen('BUILT/DATASET/DIR/')
lcnn = LightCNN(classes=datagen.get_classes())
train_gen = datagen.get_generator('train', batch_size=64)

lcnn.train(train_gen=train_gen, valid_gen=train_gen,
           optimizer=SGD(lr=0.001, momentum=0.9, decay=0.00004, nesterov=True),
           classifier_dropout=0.7, steps_per_epoch=1000, validation_steps=100,
           epochs=500, out_prefix="SOME/PREFIX_", out_period=5)

lcnn.train(train_gen=train_gen, valid_gen=train_gen,
           optimizer=SGD(lr=0.0001, momentum=0.9, decay=0.00004, nesterov=True),
           classifier_dropout=0.5, steps_per_epoch=1000, validation_steps=100,
           epochs=500, out_prefix="SOME/PREFIX_", out_period=5)
```

I trained twice as above. Totally, ran 64 batch_size x 1000 steps x (500 + 500) epochs. It takes about 5 days with GeForce GTX 1080 Ti. I used train dataset not only training but also  validation to check the score without dropout and it recorded 98.9% accuracy on validation.

## Evaluation

```python
from light_cnn import LightCNN
import celeb_gen

datagen = celeb_gen.Datagen('BUILT/DATASET/DIR/')
lcnn = LightCNN(classes=datagen.get_classes(),
                extractor_weights='TRAINED/EXTRACTOR/WEIGHTS.hdf5',
                classifier_weights='TRAINED/CLASSIFIER/WEIGHTS.hdf5')

test_gen = datagen.get_generator('test', batch_size=64)
score = lcnn.evaluate(test_gen, steps=100)
print(score)
```
I did easy evaluation as above. I evaluated only with pre-split valid set from the MS-Celeb-1M (described as in author's paper) and it recorded 95.5% accuracy.
Not yet evaluated enough on other dataset.


## Classify
```python
from light_cnn import LightCNN
import celeb_gen
import numpy as np             

datagen = celeb_gen.Datagen('BUILT/DATASET/DIR/')
lcnn = LightCNN(classes=datagen.get_classes(),
                extractor_weights='TRAINED/EXTRACTOR/WEIGHTS.hdf5',
                classifier_weights='TRAINED/CLASSIFIER/WEIGHTS.hdf5')


gen = datagen.get_generator('test', batch_size=10)
x, y = next(gen)
y = np.argmax(y, axis=-1)

y_pred = lcnn.predict_class(x)

print('y')
print(y)

print('y_pred')
print(y_pred)
```

## Extract features
```python
from light_cnn import LightCNN
import celeb_gen

datagen = celeb_gen.Datagen('BUILT/DATASET/DIR/')
lcnn = LightCNN(classes=datagen.get_classes(),
                extractor_weights='TRAINED/EXTRACTOR/WEIGHTS.hdf5')

gen= datagen.get_generator('test')
x, _ = next(gen)

feat = lcnn.exract_features(x)

print(feat)
```
