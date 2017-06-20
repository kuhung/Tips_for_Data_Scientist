# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Input
import scipy
from sklearn.metrics import fbeta_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print 'model loaded.'
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

n_classes = 3


train_path = "../img/train/"
test_path = "../img/test/"
train = pd.read_csv("../train_label.csv")
test = pd.read_csv("../test_stg1_label.csv")

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train['label'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# use ResNet50 model extract feature from fc1 layer
base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
input = Input(shape=(224,224,3),name = 'image_input')
x = base_model(input)
x = Flatten()(x)
model = Model(inputs=input, outputs=x)

X_train = []
y_train = []

for f, tags in tqdm(train.values[:], miniters=1000):
    img_path = train_path + "{}".format(f)
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)
        features_reduce =  features.squeeze()
        X_train.append(features_reduce)

        targets = np.zeros(n_classes)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
            y_train.append(targets)
    except:
        print '%s File broken'%img_path

X = np.array(X_train)
y = np.array(y_train, np.uint8)

X_test = []

for f, tags in tqdm(test.values[:], miniters=1000):
    img_path = test_path + "{}".format(f)
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)
        features_reduce =  features.squeeze()
        X_test.append(features_reduce)
    except:
        print '%s File broken'%img_path
            
            
train_ResNet =  pd.DataFrame(np.array(X_train))
train_ResNet = pd.concat([train, train_ResNet], axis = 1)
train_ResNet.to_csv('output/train_ResNet_Feature.csv',index=False)

test_ResNet =  pd.DataFrame(np.array(X_test))
test_ResNet = pd.concat([test, test_ResNet], axis = 1)
test_ResNet.to_csv('output/test_ResNet_Feature.csv',index=False)
