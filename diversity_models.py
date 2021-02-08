# diversity model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l2
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling2D

weight_decay = 0.0005



class H_Model(keras.Model):
    def __init__(self):
        super(H_Model, self).__init__()
        self.conv1 = Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.mp1 = MaxPooling2D(pool_size=2, strides=2, padding='valid')

        self.conv2 = Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.mp2 = MaxPooling2D(pool_size=2, strides=2, padding='valid')

        self.conv3 = Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.mp3 = MaxPooling2D(pool_size=2, strides=2, padding='valid')

        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu', kernel_initializer='he_normal')
        self.dp = Dropout(0.2)
        self.d2 = Dense(10, activation='softmax', kernel_initializer='he_normal')

    def call(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dp(x)
        return self.d2(x)


def H_module(input_dim):
    x = keras.Input(shape=input_dim)
    z = Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')(x)
    z = Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(z)

    z = Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')(z)
    z = Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(z)

    z = Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')(z)
    z = Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')(z)
    z = GlobalMaxPooling2D()(z)

    z = Flatten()(z)

    z = Dropout(0.5)(z)
    lable_out = Dense(10, activation='softmax')(z)

    model = keras.Model(inputs=x, outputs=lable_out)
    return model


class mul_layer(keras.layers.Layer):
    def __init__(self, input_dim=2):
        super(mul_layer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim,), dtype="float32"), trainable=True)

    def call(self, inputs):
        # return tf.matmul(inputs[0], self.w[0])+tf.matmul(inputs[1], self.w[1])
        return self.w[0] * inputs[0] + self.w[1] * inputs[1]


def Evaluator_module(input_dim):
    f0 = keras.Input(shape=input_dim)
    f1 = keras.Input(shape=input_dim)
    cf = mul_layer(2)([f0, f1])
    label_out = layers.LeakyReLU()(cf)
    # label_out = keras.layers.LayerNormalization(axis=-1)(cf)
    model = keras.Model(inputs=[f0, f1], outputs=label_out)
    return model


def Evaluator_dense_module(input_dim):
    f0 = keras.Input(shape=input_dim)
    f1 = keras.Input(shape=input_dim)
    cf0 = keras.layers.Dense(input_dim, activation='relu')(f0)
    cf1 = keras.layers.Dense(input_dim, activation='relu')(f1)
    cf = mul_layer(2)([cf0, cf1])
    label_out = layers.Softmax()(cf)
    # cf0=tf.expand_dims(f0,axis=1)
    # cf1=tf.expand_dims(f1,axis=1)
    # cross_feature=keras.layers.concatenate([cf0,cf1],axis=1)
    # cf=tf.transpose(cross_feature, perm=[0, 2, 1])
    # cf=keras.layers.Dense(2,activation='relu')(cf)
    # cf=tf.transpose(cf, perm=[0, 2, 1])
    # label_out=keras.layers.Dense(10,activation='softmax')(cf)
    # label_out=tf.reshape(label_out,[64,10])
    model = keras.Model(inputs=[f0, f1], outputs=label_out)
    return model