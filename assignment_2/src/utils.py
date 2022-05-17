import os
import urllib.request as http
from zipfile import ZipFile

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import save_model, load_model


def load_cifar10(num_classes=3):
    """
    Downloads CIFAR-10 dataset, which already contains a training and test set,
    and return the first `num_classes` classes.
    Example of usage:

    >>> (x_train, y_train), (x_test, y_test) = load_cifar10()

    :param num_classes: int, default is 3 as required by the assignment.
    :return: the filtered data.
    """
    (x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()

    fil_train = tf.where(y_train_all[:, 0] < num_classes)[:, 0]
    fil_test = tf.where(y_test_all[:, 0] < num_classes)[:, 0]

    y_train = y_train_all[fil_train]
    y_test = y_test_all[fil_test]

    x_train = x_train_all[fil_train]
    x_test = x_test_all[fil_test]

    return (x_train, y_train), (x_test, y_test)


def make_dataset(imgs, labels, label_map, img_size, rgb=True, keepdim=True, shuffle=True):
    x = []
    y = []
    n_classes = len(list(label_map.keys()))
    for im, l in zip(imgs, labels):
        # preprocess img
        x_i = im.resize(img_size)
        if not rgb:
            x_i = x_i.convert('L')
        x_i = np.asarray(x_i)
        if not keepdim:
            x_i = x_i.reshape(-1)
        
        # encode label
        y_i = np.zeros(n_classes)
        y_i[label_map[l]] = 1.
        
        x.append(x_i)
        y.append(y_i)
    x, y = np.array(x).astype('float32'), np.array(y)
    if shuffle:
        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        x, y = x[idxs], y[idxs]
    return x, y


def load_images(path):
    img_files = os.listdir(path)
    imgs, labels = [], []
    for i in img_files:
        if i.endswith('.jpg'):
            # load the image (here you might want to resize the img to save memory)
            imgs.append(Image.open(os.path.join(path, i)).copy())
    return imgs


def load_images_with_label(path, classes):
    imgs, labels = [], []
    for c in classes:
        # iterate over all the files in the folder
        c_imgs = load_images(os.path.join(path, c))
        imgs.extend(c_imgs)
        labels.extend([c] * len(c_imgs))
    return imgs, labels


def save_keras_model(model, filename):
    """
    Saves a Keras model to disk.
    Example of usage:

    >>> model = Sequential()
    >>> model.add(Dense(...))
    >>> model.compile(...)
    >>> model.fit(...)
    >>> save_keras_model(model, 'my_model.h5')

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    save_model(model, filename)


def load_keras_model(filename):
    """
    Loads a compiled Keras model saved with models.save_model.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = load_model(filename)
    return model
