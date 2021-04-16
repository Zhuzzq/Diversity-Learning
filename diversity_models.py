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


def expand_conv(init, base, k, stride):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    shortcut = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    shortcut = Activation('relu')(shortcut)

    x = ZeroPadding2D((1, 1))(shortcut)
    x = Convolution2D(base * k, (3, 3), strides=stride, padding='valid', kernel_initializer='he_normal', use_bias=False)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(base * k, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)

    # Add shortcut

    shortcut = Convolution2D(base * k, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)

    m = Add()([x, shortcut])

    return m


def conv_block(input, n, stride, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(n * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(n * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    m = Add()([init, x])
    return m


def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Wide Residual Network with specified parameters
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    ip = Input(shape=input_dim)

    x = ZeroPadding2D((1, 1))(ip)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    nb_conv = 4

    x = expand_conv(x, 16, k, stride=(1, 1))

    for i in range(N - 1):
        x = conv_block(x, n=16, stride=(1, 1), k=k, dropout=dropout)
        nb_conv += 2

    x = expand_conv(x, 32, k, stride=(2, 2))

    for i in range(N - 1):
        x = conv_block(x, n=32, stride=(2, 2), k=k, dropout=dropout)
        nb_conv += 2

    x = expand_conv(x, 64, k, stride=(2, 2))

    for i in range(N - 1):
        x = conv_block(x, n=64, stride=(2, 2), k=k, dropout=dropout)
        nb_conv += 2

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    if verbose:
        print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


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


class betterH1(keras.Model):
    def __init__(self):
        super(betterH1, self).__init__()
        self.conv1_1 = Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.conv1_2 = Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.mp1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')

        self.conv2_1 = Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.conv2_2 = Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.mp2 = MaxPooling2D(pool_size=2, strides=2, padding='valid')

        self.conv3_1 = Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.conv3_2 = Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_uniform')
        self.mp3 = GlobalMaxPooling2D()

        self.flatten = Flatten()
        # self.d1 = Dense(256, activation='relu')
        self.dp = Dropout(0.5)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.mp1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.mp2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.mp3(x)

        x = self.flatten(x)
        # x = self.d1(x)
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
    lable_out = Dense(10, activation='softmax', kernel_initializer='he_normal')(z)

    model = keras.Model(inputs=x, outputs=lable_out)
    return model


class better0_H(keras.Model):
    def __init__(self):
        super(better0_H, self).__init__()
        self.conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')
        self.bn2 = layers.BatchNormalization()
        self.mp2 = layers.MaxPool2D((2, 2))
        self.dp2 = layers.Dropout(0.2)

        self.conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')
        self.bn4 = layers.BatchNormalization()
        self.mp4 = layers.MaxPool2D((2, 2))
        self.dp4 = layers.Dropout(0.3)

        self.conv5 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')
        self.bn5 = layers.BatchNormalization()
        self.conv6 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')
        self.bn6 = layers.BatchNormalization()
        self.mp6 = layers.MaxPool2D((2, 2))
        self.dp6 = layers.Dropout(0.4)

        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu', kernel_initializer='he_uniform')
        self.bn7 = layers.BatchNormalization()
        self.dp7 = layers.Dropout(0.5)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.dp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.mp4(x)
        x = self.dp4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = selx = self.bn6(x)
        x = self.mp6(x)
        x = self.dp6(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn7(x)
        x = self.dp7(x)
        return self.d2(x)


class mul_layer(keras.layers.Layer):
    def __init__(self, input_dim=2):
        super(mul_layer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim,), dtype="float32"), trainable=True)

    def call(self, inputs):
        # return tf.matmul(inputs[0], self.w[0])+tf.matmul(inputs[1], self.w[1])
        return (self.w[0] * inputs[0] + self.w[1] * inputs[1])


class weight_layer(keras.layers.Layer):
    def __init__(self, input_dim=2):
        super(weight_layer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim,), dtype="float32"), trainable=True)

    def call(self, inputs):
        # return tf.matmul(inputs[0], self.w[0])+tf.matmul(inputs[1], self.w[1])
        return (self.w[0] * inputs[0] + self.w[1] * inputs[1])


def Evaluator_module(input_dim):
    f0 = keras.Input(shape=input_dim)
    f1 = keras.Input(shape=input_dim)
    cf = mul_layer(2)([f0, f1])
    label_out = layers.ReLU()(cf)
    # label_out = keras.layers.LayerNormalization(axis=-1)(cf)
    label_out = label_out / tf.reduce_sum(label_out, 1, keepdims=True)
    model = keras.Model(inputs=[f0, f1], outputs=label_out)
    return model


def weighting_module(input_dim):
    f0 = keras.Input(shape=input_dim)
    f1 = keras.Input(shape=input_dim)
    cf = weight_layer(2)([f0, f1])
    label_out = layers.ReLU()(cf)
    # label_out = keras.layers.LayerNormalization(axis=-1)(cf)
    label_out = label_out / tf.reduce_sum(label_out, 1, keepdims=True)
    model = keras.Model(inputs=[f0, f1], outputs=label_out)
    return model


def dense_ensemble(input_dim):
    f0 = keras.Input(shape=input_dim)
    f1 = keras.Input(shape=input_dim)
    f = tf.concat([f0, f1], 1)
    cf = keras.layers.Dense(input_dim, activation='relu')(f)
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
# resnet layer


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    # ResNet Version 1 Model builder [a]
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
