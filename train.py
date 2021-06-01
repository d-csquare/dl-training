import os
import platform
import numpy as np
import pandas as pd
import random as python_random
import matplotlib.pyplot as plt
from IPython.display import clear_output

os.environ['PYTHONHASHSEED'] = '73'
python_random.seed(73)
np.random.seed(73)

print(platform.platform())
%matplotlib inline

### TPU Check
import tensorflow as tf
tf.random.set_seed(73)
TPU_INIT = False

if TPU_INIT:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime!')
else:
    !nvidia-smi

clear_output()
print("Tensorflow version " + tf.__version__)

### Check Data source and change....
MyDrive = '/kaggle/working'
clear_output()

DataDir = '/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset'
os.listdir(DataDir)


import cv2
import time
import imageio
import imgaug.augmenters as iaa
import imgaug as ia
from tqdm import tqdm
ia.seed(73)

IMG_SIZE = 224
ColorCh = 3

CATEGORIES = []
for list_ in os.listdir(DataDir):
    if not '.' in list_:
        CATEGORIES.append(list_)

print(CATEGORIES)

def isValid(text):
    supported_types = ['.png', '.jpg', '.jpeg']
    for img_type in supported_types:
        if img_type in text:
            return True
        else:
            return False

import random
random.seed(73)

def prepareData(Dir, split_ratio):
    X = []
    y = []
    Frame = []

    flip = iaa.Fliplr(1.0)
    zoom = iaa.Affine(scale=1)
    random_brightness = iaa.Multiply((1, 1.2))
    rotate = iaa.Affine(rotate=(-20, 20))

    for i, category in enumerate(CATEGORIES):
        path = os.path.join(Dir, category, (category))
        if not os.path.isdir(path):
            pass

        else:
            class_num = CATEGORIES.index(category)
            limit = 500 # images from each class
            img_list = os.listdir(path)[0:limit]
            random.shuffle(img_list)

            for img in tqdm(img_list):
                if isValid(img):
                    orig_img = cv2.imread(os.path.join(path,img) , cv2.IMREAD_COLOR)
                    image_aug = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE),
                                           interpolation = cv2.INTER_CUBIC)

                    image_aug = flip(image = image_aug)
                    image_aug = random_brightness(image = image_aug)
                    image_aug = zoom(image = image_aug)
                    image_aug = rotate(image = image_aug)

                    image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                    X.append(image_aug)
                    y.append(class_num)
                else:
                    pass

    if len(X) > 0:
        train_len = int(len(X) * split_ratio)

        features = pd.DataFrame((np.array(X)).reshape(-1, IMG_SIZE * IMG_SIZE * ColorCh))
        labels = pd.DataFrame({'label': y})

        Frame = pd.concat([features, labels], axis=1).sample(frac = 1, random_state=73)
        train_df, test_df = Frame[train_len:], Frame[:train_len]

        return train_df, test_df

    train_df, test_df = prepareData(DataDir, split_ratio=0.2)

    import matplotlib.pyplot as plt

def print_images(samples):
    images = samples.drop(["label"],axis = 1).to_numpy()
    labels = samples['label'].to_numpy()

    fig=plt.figure(figsize=(20, 8))
    columns = 4
    rows = 1

    for i, image in enumerate(images):
        fig.add_subplot(rows,columns,i + 1)
        title = '{}'.format(CATEGORIES[labels[i]])
        Sample_image = image.reshape(IMG_SIZE, IMG_SIZE, ColorCh)
        plt.imshow(Sample_image, cmap='gray')
        plt.title(title)

    plt.show()

label_count = pd.Series(test_df['label'].values.ravel()).value_counts()
for label in list(label_count.index.values):
    samples = (test_df[test_df['label']==int(label)].iloc[0:4])
    print_images(samples)

%matplotlib inline

print('> DataFrame shape: ',train_df.shape)
print('> {} image data '.format(train_df.shape[0]))
print('> {} --> ({} * {} * {}) pixels + 1 label for each image (1d scaled)\n'.format(train_df.shape[1], IMG_SIZE,IMG_SIZE, ColorCh))

label_count_train = pd.Series(train_df['label'].values.ravel()).value_counts()
n_classes = len(label_count_train)
print('> Total Classes: {}'.format(n_classes))

print('> label count for train')
for i in range(len(label_count_train)):
    print('> {} : {}'.format(CATEGORIES[i], label_count_train[i]))


