from cifar10 import Cifar10
import tensorflow as tf
from tensorflow.python.keras import metrics

# https://keras.io/api/metrics/
METRICS = [
    metrics.CategoricalAccuracy(),
    metrics.Precision(),
    metrics.Recall(),
]


def model_example():
    #https://www.kaggle.com/code/amyjang/tensorflow-cifar10-cnn-tutorial/notebook

    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu'),
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
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

    print(model.summary())

    #https://keras.io/api/losses/
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
            loss='categorical_crossentropy', metrics=METRICS)

    return model

def model_sigmoid():
    # Igual ao model_example, somente trocando a função de ativação para sigmoid
    # Com sigmoid, o modelo não performa bem, acurácia fica baixa
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='sigmoid'),
            tf.keras.layers.Conv2D(32, 3, activation='sigmoid'),
            tf.keras.layers.MaxPooling2D(),
            # tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='sigmoid'),
            tf.keras.layers.Conv2D(64, 3, activation='sigmoid'),
            tf.keras.layers.MaxPooling2D(),
            # tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='sigmoid'),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

    print(model.summary())

    #https://keras.io/api/losses/
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
            loss='categorical_crossentropy', metrics=METRICS)

    return model

def model_vgg19():
    #https://www.kaggle.com/code/adi160/cifar-10-keras-transfer-learning/notebook
    pass

def model_resnet():
    #https://www.kaggle.com/code/adi160/cifar-10-keras-transfer-learning/notebook
    pass

if(__name__ == "__main__"):
    cifar = Cifar10()
    #cifar.run()
    cifar.run_training(model=model_example(), epochs = 2, batch_size = 256, shuffle=True)