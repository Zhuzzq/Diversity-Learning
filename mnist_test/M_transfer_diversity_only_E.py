# transfer learning using diversity structure based on the pre-trained models, only E-model will be trained.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
import datetime
from M_diversity_models import M_H_module, Evaluator_module, Evaluator_dense_module, weighting_module, dense_ensemble
from keras.preprocessing.image import ImageDataGenerator
import sys


print("TensorFlow version: ", tf.__version__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# load data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0

x_train = train_images
x_test = test_images

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

batch_size = 16
train_ds = tf.data.Dataset.from_tensor_slices((x_train, train_labels)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, test_labels)).batch(batch_size)

print(train_ds)


# learning configures
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(amsgrad=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

tf.random.set_seed(7)
# load models
# H0 = keras.models.load_model('models/ori/20210202-122602/simple_model.005.h5')
# H1 = keras.models.load_model('models/ori/20210202-103441/simple_model.005.h5')
# H0 = H_Model()
# H1 = H_Model()
# H0 = keras.models.load_model('logs/models/ori/h_epoch1000_20210110-223807')
# H1 = keras.models.load_model('logs/models/ori/h_epoch1000_20210110-223807')
trans = int(sys.argv[1])
input_shape = (28, 28, 1)
if trans:
    # H0 = resnet_v1(input_shape=input_shape, depth=20)
    # H1 = resnet_v1(input_shape=input_shape, depth=20)
    # H0.load_weights('models/ori/20210412-213315/cifar10_ResNet20v1_model.070.h5')
    # H1.load_weights('models/ori/20210412-213315/cifar10_ResNet20v1_model.071.h5')
    # H0 = keras.models.load_model('models/ori/20210413-184250/h_epoch10941')
    # H1 = keras.models.load_model('models/ori/20210413-173316/h_epoch10941')
    H0 = keras.models.load_model('models/ori/20211009-094458/h_epoch16002')
    H1 = keras.models.load_model('models/ori/20211009-103109/h_epoch16002')
    # H0 = keras.models.load_model('models/ori/20210413-220920/h_epoch9378')
    # H1 = keras.models.load_model('models/ori/20210413-232311/h_epoch9378')
    # H0 = keras.models.load_model('models/ori/20210204-150510/h_epoch9378')
    # H1 = keras.models.load_model('models/ori/20210204-153931/h_epoch9378')
else:
    H0 = M_H_module(input_shape)
    H1 = M_H_module(input_shape)

cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

if sys.argv[3] == 'ED':
    E0 = Evaluator_dense_module((10))
    keras.utils.plot_model(E0, 'Ed.png', show_shapes=True)
    E0.summary()
    train_log_dir = '../logs/fit/' + cur_time + 'trans_ED'
    summary_writer = tf.summary.create_file_writer(train_log_dir)
elif sys.argv[3] == 'W':
    E0 = weighting_module((10))
    keras.utils.plot_model(E0, 'W.png', show_shapes=True)
    E0.summary()
    train_log_dir = '../logs/fit/' + cur_time + 'trans_W'
    summary_writer = tf.summary.create_file_writer(train_log_dir)
elif sys.argv[3] == 'WD':
    E0 = dense_ensemble((10))
    keras.utils.plot_model(E0, 'Wd.png', show_shapes=True)
    E0.summary()
    train_log_dir = '../logs/fit/' + cur_time + 'trans_WD'
    summary_writer = tf.summary.create_file_writer(train_log_dir)
else:
    E0 = Evaluator_module((10))
    keras.utils.plot_model(E0, 'E.png', show_shapes=True)
    E0.summary()

    train_log_dir = '../logs/fit/' + cur_time + 'trans_E'
    summary_writer = tf.summary.create_file_writer(train_log_dir)

# train & test


@tf.function
def transfer_diversity_train(images, labels):
    with tf.GradientTape(persistent=True) as tape:
        r0 = H0(images[0]) + H1(images[1])
        r1 = -H0(images[1]) + H1(images[0])
        s0 = E0([r0, r1])
        s1 = E0([-r1, r0])
        loss = loss_object(labels[0], s0) / 2 + loss_object(labels[1], s1) / 2
    E0_gradients = tape.gradient(loss, E0.trainable_variables)
    # H0_gradients = tape.gradient(loss, H0.trainable_variables)
    # H1_gradients = tape.gradient(loss, H1.trainable_variables)
    optimizer.apply_gradients(zip(E0_gradients, E0.trainable_variables))
    # optimizer.apply_gradients(zip(H0_gradients, H0.trainable_variables))
    # optimizer.apply_gradients(zip(H1_gradients, H1.trainable_variables))
    del tape

    train_loss(loss)
    train_accuracy.update_state(labels[0], s0)
    train_accuracy.update_state(labels[1], s1)


@tf.function
def diversity_test(images, labels):
    r0 = H0(images[0]) + H1(images[1])
    r1 = -H0(images[1]) + H1(images[0])
    s0 = E0([r0, r1])
    s1 = E0([-r1, r0])
    t_loss = loss_object(labels[0], s0) / 2 + loss_object(labels[1], s1) / 2

    test_loss(t_loss)
    test_accuracy.update_state(labels[0], s0)
    test_accuracy.update_state(labels[1], s1)


# training settings
train_loss.reset_states()
train_accuracy.reset_states()
test_loss.reset_states()
test_accuracy.reset_states()


# train
total_epoch = 0
max_episode = int(sys.argv[2])
H0.save('models/trans_only_E/{}/h0'.format(cur_time))
H1.save('models/trans_only_E/{}/h1'.format(cur_time))


train_l = list(train_ds)

test_l = list(test_ds)
print('train size: {}\ntest size:{}'.format(len(train_l), len(test_l)))

for episode in range(max_episode):
    epoch = 0
    while epoch < len(train_l) - 1:
        images0, labels0 = train_l[epoch]
        images1, labels1 = train_l[epoch + 1]
        images = [images0, images1]
        labels = [labels0, labels1]
        transfer_diversity_train(images, labels)

        with summary_writer.as_default():
            tf.summary.scalar('Main/loss', train_loss.result(), step=total_epoch)
            tf.summary.scalar('Main/acc', train_accuracy.result() * 100, step=total_epoch)
        summary_writer.flush()
        epoch += 2
        total_epoch += 2
        if total_epoch % 400 == 2:
            k = 0
            while k < len(test_l) - 1:
                t_images0, t_labels0 = test_l[k]
                t_images1, t_labels1 = test_l[k + 1]
                test_images = [t_images0, t_images1]
                test_labels = [t_labels0, t_labels1]
                k += 2
                diversity_test(test_images, test_labels)
            with summary_writer.as_default():
                tf.summary.scalar('Main/t_loss', test_loss.result(), step=total_epoch)
                tf.summary.scalar('Main/t_acc', test_accuracy.result() * 100, step=total_epoch)
            summary_writer.flush()
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(total_epoch, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))

    E0.save('models/trans_only_E/{}/e0_epoch_{}'.format(cur_time, total_epoch))
