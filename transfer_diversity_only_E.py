# transfer learning using diversity structure based on the pre-trained models, only E-model will be trained.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
import datetime
from diversity_models import H_Model, Evaluator_module, Evaluator_dense_module
from keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version: ", tf.__version__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# load data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


x_train = train_images
x_test = test_images

batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((x_train, train_labels)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, test_labels)).batch(batch_size)

print(train_ds)


# learning configures
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


# load models
H0 = keras.models.load_model('models/ori/20210204-150510/h_epoch3126')
H1 = keras.models.load_model('models/ori/20210204-153931/h_epoch3126')
E0 = Evaluator_module((10))
keras.utils.plot_model(E0, 'E.png', show_shapes=True)
E0.summary()


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
    optimizer.apply_gradients(zip(E0_gradients, E0.trainable_variables))
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


cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = '../logs/fit/' + cur_time + 'trans_E'
summary_writer = tf.summary.create_file_writer(train_log_dir)


# train
total_epoch = 0
max_episode = 25
H0.save('models/trans_only_E/{}/h0'.format(cur_time))
H1.save('models/trans_only_E/{}/h1'.format(cur_time))


data_augmentation = False
if data_augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )
    datagen.fit(x_train)
    train_ds = datagen.flow(x_train, train_labels, batch_size=batch_size)
    print('after aug:{}'.format(len(train_ds)))


train_l = list(train_ds)
test_l = list(test_ds)
print('train size: {}\n test size:{}'.format(len(train_l), len(test_l)))

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
