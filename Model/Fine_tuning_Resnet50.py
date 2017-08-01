'''
 Created on: 2017/08/01
 Author: kuhung
 Refer:
 https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/30054
 https://github.com/flyyufelix/cnn_finetune
'''



import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input,pooling
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.applications.vgg19 import VGG19

from keras.applications.resnet50 import ResNet50
from keras.models import Model

# get tags
col = ['id','tags','source']

train = pd.read_csv('../input/data_train_image.txt',sep=' ',header=None)
train.columns = col
train.drop(['source'],axis=1,inplace=True)

val = pd.read_csv('../input/val.txt',sep=' ',header=None)
val.columns = col
val.drop(['source'],axis=1,inplace=True)
train=pd.concat([train,val])

test = pd.read_csv('../input/test_1.txt',sep=' ',header=None)
test.columns=['id']
test['tags']=0
tag_file = train.tags.unique()

import cv2
from tqdm import tqdm

# Params
input_size = 224
input_channels = 3

epochs = 80
batch_size = 32
learning_rate = 0.001
lr_decay = 0.002

valid_data_size = 0.3  # Samples to withhold for validation


input_tensor = Input(shape=(input_size,input_size,input_channels)) 
base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

#x = base_model.output
x = base_model.get_layer(index = -1).output
#x = pooling.AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
'''
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
'''
predictions = Dense(100, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers


'''
for layer in base_model.layers:
    layer.trainable = True


for layer in base_model.layers[:25]:
    layer.trainable = False
'''

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}


x_test = []
for f,tags in tqdm(test.values, miniters=100):
    img = cv2.resize(cv2.imread('../input/test_1/image/{}.jpg'.format(f)), (input_size, input_size))
    x_test.append(img)

x_test = np.array(x_test, np.float32)
x_test/=255


df_train,df_valid = train_test_split(train, test_size=valid_data_size, random_state=42)

x_valid = []
y_valid = []

for f, tags in tqdm(df_valid.values, miniters=100):
    img = cv2.resize(cv2.imread('../input/train/{}.jpg'.format(f)), (input_size, input_size))
    targets = np.zeros(100)
    targets[label_map[tags]] = 1
    x_valid.append(img)
    y_valid.append(targets)

y_valid = np.array(y_valid, np.uint8)
x_valid = np.array(x_valid, np.float32)
x_valid/=255


x_train = []
y_train = []


for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.resize(cv2.imread('../input/train/{}.jpg'.format(f)), (input_size, input_size))
    targets = np.zeros(100)
    targets[label_map[tags]] = 1
    
    x_train.append(img)
    y_train.append(targets)

    img = cv2.flip(img, 1)  # flip horizontally
    x_train.append(img)
    y_train.append(targets)

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)
x_train/=255

bing_normal_1=np.vstack((x_train,x_valid))
bing_normal_2=np.vstack((bing_normal_1,x_test))

x_test -= np.mean(bing_normal_2, axis = 0)
#x_test /= np.std(bing_normal_2, axis = 0)

x_valid -= np.mean(bing_normal_2, axis = 0)
#x_valid /= np.std(bing_normal_2, axis = 0)

x_train -= np.mean(bing_normal_2, axis = 0)
#x_train /= np.std(bing_normal_2, axis = 0)


callbacks = [EarlyStopping(monitor='val_acc',
                           patience=5,
                           verbose=0),
             TensorBoard(log_dir='logs'),
             ModelCheckpoint('weights_resnet50.h5',
                             save_best_only=True)]

#opt = Adam(lr=learning_rate, decay=lr_decay)
opt = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1)

validation_generator = test_datagen.flow(x_valid,y_valid,seed=42)

#model.load_weights('weights.h5')

model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=test_datagen.flow(x_valid,y_valid),
                    validation_steps=64,
                    callbacks=callbacks,
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

prediction=model.predict(x_test)
prediction_index = prediction.argmax(axis=1)
prediction_tags=tag_file[prediction_index]

output=pd.DataFrame({'0_tags':prediction_tags,'1_id':test.id.values})
output.to_csv('../output/baseline_resnet50lre3dec2e3.txt',header=None,index=False,sep='\t')

