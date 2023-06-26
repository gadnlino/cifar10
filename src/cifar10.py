#https://keras.io/getting_started/intro_to_keras_for_engineers/


import os
import pickle

import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from datetime import datetime

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

RESULTS_FOLDER = f'files/results/{str(datetime.utcnow().timestamp()).replace(".", "")}'

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

class Cifar10:
    def run_training(self, model, epochs = 10, batch_size = 32,shuffle=False):
        print(f'Results will be saved at the folder : {RESULTS_FOLDER}')
        #https://www.kaggle.com/code/amyjang/tensorflow-cifar10-cnn-tutorial/notebook

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        y_train = y_train.flatten()
        y_test = y_test.flatten()

        #transformando x_train e x_test em 1 tensor com dimensão 50000x32x32x3 cada
        x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

        #min-max normalization
        x_train=x_train / 255.0
        x_test=x_test / 255.0

        #realizando encoding onehot dos labels 
        y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
        y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

        # plt.imshow(x_train[100])
        # plt.savefig(f"{RESULTS_FOLDER}/100_train_example.jpg")
        # plt.show(block=False)
        # plt.clf()

        # print(y_train[100])

        #https://keras.io/api/models/model_training_apis/
        history = model.fit(\
            x_train, y_train, batch_size=batch_size, epochs=epochs, \
            use_multiprocessing=True, validation_data = (x_test, y_test),\
                shuffle=shuffle)        

        print(history.history)

        pd.DataFrame(data=history.history, columns=history.history.keys()).to_csv(f"{RESULTS_FOLDER}/history.csv")

        fig, ax = plt.subplots(2,2)
        ax[0][0].plot(history.history['loss'], color='b', label="Training Loss")
        ax[0][0].plot(history.history['val_loss'], color='r', label="Validation Loss")
        legend = ax[0][0].legend(loc='best', shadow=True)

        ax[0][1].plot(history.history['categorical_accuracy'], color='b', label="Training Accuracy")
        ax[0][1].plot(history.history['val_categorical_accuracy'], color='r', label="Validation Accuracy")
        legend = ax[0][1].legend(loc='best', shadow=True)

        ax[1][0].plot(history.history['precision'], color='b', label="Training Precision")
        ax[1][0].plot(history.history['val_precision'], color='r', label="Validation Precision")
        legend = ax[1][0].legend(loc='best', shadow=True)

        ax[1][1].plot(history.history['recall'], color='b', label="Training Recall")
        ax[1][1].plot(history.history['val_recall'], color='r', label="Validation Recall")
        legend = ax[1][1].legend(loc='best', shadow=True)

        fig.savefig(f"{RESULTS_FOLDER}/metrics.jpg")
        ax[0][0].clear()
        ax[0][1].clear()
        ax[1][0].clear()
        ax[1][1].clear()
        fig.clf()

        y_pred = model.predict(x_test)
        # Convert predictions classes to one hot vectors 
        y_pred_classes = np.argmax(y_pred, axis = 1) 
        # Convert validation observations to one hot vectors
        y_true = np.argmax(y_test,axis = 1)
        # compute the confusion matrix
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)

        plt.figure(figsize=(12, 9))
        c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
        c.set(xticklabels=CLASSES, yticklabels=CLASSES)
        plt.savefig(f"{RESULTS_FOLDER}/confusion_mtx.jpg")
        plt.clf()