from cifar10 import Cifar10
import tensorflow as tf
from tensorflow.python.keras import metrics
from keras.applications import VGG19, ResNet50

# https://keras.io/api/metrics/
METRICS = [
    metrics.CategoricalAccuracy(),
    metrics.Precision(),
    metrics.Recall(),
]

INPUT_SHAPE = (32, 32, 3)


def model_example():
    #https://www.kaggle.com/code/amyjang/tensorflow-cifar10-cnn-tutorial/notebook

    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=INPUT_SHAPE, activation='relu'),
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

    #https://keras.io/api/losses/
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
            loss='categorical_crossentropy', metrics=METRICS)

    return model

def model_sigmoid():
    # Igual ao model_example, somente trocando a função de ativação para sigmoid
    # Com sigmoid, o modelo não performa bem, acurácia fica baixa
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=INPUT_SHAPE, activation='sigmoid'),
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

    #https://keras.io/api/losses/
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
            loss='categorical_crossentropy', metrics=METRICS)

    return model

def model_vgg19():
    #https://www.kaggle.com/code/adi160/cifar-10-keras-transfer-learning/notebook
    
    base_model_1 = VGG19(include_top=False,weights='imagenet',input_shape=INPUT_SHAPE,classes=10)

    model_1= tf.keras.models.Sequential()
    model_1.add(base_model_1) #Adds the base model (in this case vgg19 to model_1)
    model_1.add(tf.keras.layers.Flatten())
    model_1.add(tf.keras.layers.Dense(1024,activation=('relu'),input_dim=512))
    model_1.add(tf.keras.layers.Dense(512,activation=('relu'))) 
    model_1.add(tf.keras.layers.Dense(256,activation=('relu'))) 
    #model_1.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
    model_1.add(tf.keras.layers.Dense(128,activation=('relu')))
    #model_1.add(Dropout(.2))
    model_1.add(tf.keras.layers.Dense(10,activation=('softmax')))

    learn_rate=.001

    sgd=tf.keras.optimizers.SGD(lr=learn_rate,momentum=.9,nesterov=False)
    

    model_1.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=METRICS)

    return model_1

def model_resnet():
    #https://www.kaggle.com/code/adi160/cifar-10-keras-transfer-learning/notebook
    #Since we have already defined Resnet50 as base_model_2, let us build the sequential model.

    base_model_2 = ResNet50(include_top=False,weights='imagenet',input_shape=INPUT_SHAPE,classes=10)

    model_2=tf.keras.models.Sequential()
    #Add the Dense layers along with activation and batch normalization
    model_2.add(base_model_2)
    model_2.add(tf.keras.layers.Flatten())


    #Add the Dense layers along with activation and batch normalization
    model_2.add(tf.keras.layers.Dense(4000,activation=('relu'),input_dim=512))
    model_2.add(tf.keras.layers.Dense(2000,activation=('relu'))) 
    model_2.add(tf.keras.layers.Dropout(.4))
    model_2.add(tf.keras.layers.Dense(1000,activation=('relu'))) 
    model_2.add(tf.keras.layers.Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
    model_2.add(tf.keras.layers.Dense(500,activation=('relu')))
    model_2.add(tf.keras.layers.Dropout(.2))
    model_2.add(tf.keras.layers.Dense(10,activation=('softmax'))) #This is the classification layer

    learn_rate=.001

    adam=tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model_2.compile(optimizer=adam,loss='categorical_crossentropy',metrics=METRICS)

    return model_2

if(__name__ == "__main__"):
    cifar = Cifar10()
    model = model_sigmoid()    
    #cifar.run()
    cifar.run_fitting(model=model, epochs = 1, batch_size = 64, shuffle=False)