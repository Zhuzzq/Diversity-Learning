import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import datetime
import sys
from M_diversity_models import M_H_module


print("TensorFlow version: ", tf.__version__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# load data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0


print('train size: {}\n test size:{}'.format(len(train_images), len(test_images)))
print(train_images[0].size)

x_train = train_images
x_test = test_images

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

batch_size = 16
train_ds = tf.data.Dataset.from_tensor_slices((x_train, train_labels)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, test_labels)).batch(batch_size)

print(len(train_ds))


data_augmentation = int(sys.argv[1])
max_episode = int(sys.argv[2])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


input_shape = (28, 28, 1)
# h = create_wide_residual_network(init_shape, nb_classes=10, N=2, k=8, dropout=0.2)
# h.summary()
# keras.utils.plot_model(h, 'wrn.png', show_shapes=True)
# h = betterH1()
h = M_H_module(input_shape)
h.summary()

# h = resnet_v1(input_shape, 20)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = h(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, h.trainable_variables)
    optimizer.apply_gradients(zip(gradients, h.trainable_variables))

    train_loss(loss)
    train_accuracy.update_state(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = h(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy.update_state(labels, predictions)


train_loss.reset_states()
train_accuracy.reset_states()
test_loss.reset_states()
test_accuracy.reset_states()

epoch = 0


cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = '../logs/fit/' + cur_time + 'ori'
summary_writer = tf.summary.create_file_writer(train_log_dir)


for episode in range(max_episode):
    for images, labels in train_ds:
        epoch += 1
        train_step(images, labels)
        # print(epoch)
        with summary_writer.as_default():
            tf.summary.scalar('Main/loss', train_loss.result(), step=epoch)
            tf.summary.scalar('Main/acc', train_accuracy.result() * 100, step=epoch)
        summary_writer.flush()
    # for test_images, test_labels in test_ds:
    #     test_step(test_images, test_labels)
    # with summary_writer.as_default():
    #     tf.summary.scalar('Main/t_loss', test_loss.result(), step=epoch)
    #     tf.summary.scalar('Main/t_acc', test_accuracy.result() * 100, step=epoch)
    # summary_writer.flush()
    # template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    # print(template.format(epoch, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))
        if epoch % 400 == 2:
            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)
            with summary_writer.as_default():
                tf.summary.scalar('Main/t_loss', test_loss.result(), step=epoch)
                tf.summary.scalar('Main/t_acc', test_accuracy.result() * 100, step=epoch)
            summary_writer.flush()
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))
        if epoch % len(train_ds) == 0:
            break
        if epoch % 500 == 2:
            h.save('./models/ori/{}/h_epoch{}'.format(cur_time, epoch))
