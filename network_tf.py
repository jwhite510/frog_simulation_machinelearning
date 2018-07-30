import tensorflow as tf
import tables
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof
from generate_data import retrieve_data


class GetData():
    def __init__(self, batch_size):

        self.batch_counter = 0
        self.batch_index = 0
        self.batch_size = batch_size

        hdf5_file = tables.open_file("frogtrainingdata.hdf5", mode="r")
        frog = hdf5_file.root.frog[:, :]
        self.samples = np.shape(frog)[0]

        hdf5_file.close()

    def next_batch(self):

        # retrieve the next batch of data from the data source

        hdf5_file = tables.open_file("frogtrainingdata.hdf5", mode="r")
        E_real_batch = hdf5_file.root.E_real[self.batch_index:self.batch_index+self.batch_size, :]
        E_imag_batch = hdf5_file.root.E_imag[self.batch_index:self.batch_index+self.batch_size, :]
        E_appended_batch = np.append(E_real_batch, E_imag_batch, 1)
        frog_batch = hdf5_file.root.frog[self.batch_index:self.batch_index+self.batch_size, :]
        hdf5_file.close()

        self.batch_index += self.batch_size

        return  frog_batch, E_appended_batch

    def evaluation_data(self):

        # this is used to evaluate the mean squared error of the data after every epoch

        hdf5_file = tables.open_file("frogtrainingdata.hdf5", mode="r")
        E_real_eval = hdf5_file.root.E_real[:, :]
        E_imag_eval = hdf5_file.root.E_imag[:, :]
        E_appended_eval = np.append(E_real_eval, E_imag_eval, 1)
        frog_eval = hdf5_file.root.frog[:, :]
        hdf5_file.close()

        return frog_eval, E_appended_eval



def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# convolutional layer
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


# dense layer
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# placeholders
x = tf.placeholder(tf.float32, shape=[None, 6148])
y_true = tf.placeholder(tf.float32, shape=[None, 256])

# layers
x_image = tf.reshape(x, [-1, 58, 106, 1])
# shape = [sizex, sizey, channels, filters/features]
convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 3])
convo_2 = convolutional_layer(convo_1, shape=[5, 5, 3, 4])

convo_2_flat = tf.reshape(convo_2, [-1, 58*106*4])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))
full_layer_two = tf.nn.relu(normal_full_layer(full_layer_one, 1024))
y_pred = normal_full_layer(full_layer_two, 256)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 1000

# initialize data object
get_data = GetData(batch_size=10)
_, t, _, _, _, _ = retrieve_data(False, False)
plt.ion()


_, ax1 = plt.subplots(1, 2)
_, ax2 = plt.subplots(1, 2)


with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        print("Epoch : {}".format(i+1))

        # iterate through every sample in the training set
        while get_data.batch_index < get_data.samples:

            batch_x, batch_y = get_data.next_batch()
            sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

        # view the mean squared error on evaluation data
        batch_x_eval, batch_y_eval = get_data.evaluation_data()
        print("MSE: ", sess.run(loss, feed_dict={x: batch_x_eval, y_true: batch_y_eval}))
        predictions = sess.run(y_pred, feed_dict={x: batch_x_eval, y_true: batch_y_eval})

        for ax, index in zip([ax1, ax2], [1, 2]):
            ax[0].cla()
            ax[0].plot(t, predictions[index, :128], color="blue")
            ax[0].plot(t, predictions[index, 128:], color="red")
            ax[0].set_title("prediction")
            ax[1].cla()
            ax[1].plot(t, batch_y_eval[index, :128], color="blue")
            ax[1].plot(t, batch_y_eval[index, 128:], color="red")
            ax[1].set_title("actual")

        plt.show()
        plt.pause(0.001)



        # return the index to 0
        get_data.batch_index = 0









#
#

#
#


# create tensorflow model


# # open data
# hdf5_file = tables.open_file("frogtrainingdata.hdf5", mode="r")
#
# E_real = hdf5_file.root.E_real[:, :]
# fig, ax = plt.subplots(1, 1)
# ax.plot(t, E_real[0, :])
#
# E_imag = hdf5_file.root.E_imag[:, :]
# frog = hdf5_file.root.frog[:, :]
#
# hdf5_file.close()
#
# plt.show()