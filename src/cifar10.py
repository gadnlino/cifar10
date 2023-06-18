import os
import pickle

import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools



print(tf.__version__)

DATASET_FOLDER = './files/dataset'

TRAINING_FILES = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]

TEST_FILES = [
    'test_batch'
]

class Cifar10:
    def __init__(self) -> None:
        self.__model = None
        self.__label_meta = None
        self.__images_batches = []

    def __download_files(self):
        import download_files

    def __read_file(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict
    
    def __add_batch(self, dic):
        # Example of an image in a batch
        # {
        #     'label': 1,
        #     'colors': {
        #         'r': [],
        #         'g': [],
        #         'b': []
        #     }
        # }

        batch = []
        cnt = 0

        for data in dic[b'data']:
            img = {
                'label': dic[b'labels'][cnt],
                'colors_original': data,
                'colors': {
                    'r': list(data[0:1024]),
                    'g': list(data[1024:2048]),
                    'b': list(data[2048:3072]),
                }
            }

            batch.append(img)

            cnt += 1

        self.__images_batches.append(batch)

    def run(self):
        self.__download_files()

        for file in TRAINING_FILES:
            batch = self.__read_file(os.path.join(DATASET_FOLDER, file))
            self.__add_batch(batch)
        
        print(len(self.__images_batches))
        print(len(self.__images_batches[0][0]['colors']['r']))
        print(len(self.__images_batches[0][0]['colors']['g']))
        print(len(self.__images_batches[0][0]['colors']['b']))
    
    def run_tensorflow(self):
        #https://www.kaggle.com/code/amyjang/tensorflow-cifar10-cnn-tutorial/notebook

        cifar10 = tf.keras.datasets.cifar10        

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        assert x_train.shape == (50000, 32, 32, 3)
        assert x_test.shape == (10000, 32, 32, 3)
        assert y_train.shape == (50000, 1)
        assert y_test.shape == (10000, 1)

        y_train = y_train.flatten()
        y_test = y_test.flatten()

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        plt.figure(figsize=(10,7))
        p = sns.countplot(y_train.flatten())
        p.set(xticklabels=classes)

        plt.show(block=False)
        plt.clf()

        input_shape = (32, 32, 3)

        x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
        x_train=x_train / 255.0
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
        x_test=x_test / 255.0

        y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
        y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

        plt.imshow(x_train[100])
        plt.savefig("100_train_example.jpg")
        plt.show(block=False)
        plt.clf()

        print(y_train[100])

        batch_size = 32
        num_classes = 10
        epochs = 50

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
            loss='categorical_crossentropy', metrics=['acc'])

        history = model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs)

        fig, ax = plt.subplots(2,1)
        ax[0].plot(history.history['loss'], color='b', label="Training Loss")
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
        legend = ax[1].legend(loc='best', shadow=True)

        fig.savefig("loss_and_accuracy.jpg")
        ax[0].clear()
        ax[1].clear()
        fig.clf()

        y_pred = model.predict(x_test)
        # Convert predictions classes to one hot vectors 
        y_pred_classes = np.argmax(y_pred,axis = 1) 
        # Convert validation observations to one hot vectors
        y_true = np.argmax(y_test,axis = 1)
        # compute the confusion matrix
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)

        plt.figure(figsize=(12, 9))
        c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
        c.set(xticklabels=classes, yticklabels=classes)
        plt.savefig("confusion_mtx.jpg")
        plt.clf()