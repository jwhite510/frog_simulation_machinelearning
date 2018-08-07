import tensorflow as tf
import tables
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof
from generate_data import retrieve_data
import time
from frognet1 import plot_frog
import os
import random

class GetData():
    def __init__(self, batch_size):

        self.batch_counter = 0
        self.batch_index = 0
        self.batch_size = batch_size

        hdf5_file = tables.open_file("frogtrainingdata_noambiguities.hdf5", mode="r")
        frog = hdf5_file.root.frog[:, :]
        self.samples = np.shape(frog)[0]

        hdf5_file.close()

    def next_batch(self):

        # retrieve the next batch of data from the data source

        hdf5_file = tables.open_file("frogtrainingdata_noambiguities.hdf5", mode="r")
        E_real_batch = hdf5_file.root.E_real[self.batch_index:self.batch_index+self.batch_size, :]
        E_imag_batch = hdf5_file.root.E_imag[self.batch_index:self.batch_index+self.batch_size, :]
        E_appended_batch = np.append(E_real_batch, E_imag_batch, 1)
        frog_batch = hdf5_file.root.frog[self.batch_index:self.batch_index+self.batch_size, :]
        hdf5_file.close()

        self.batch_index += self.batch_size

        return  frog_batch, E_appended_batch

    def next_batch_random(self):

        # generate random indexes for the batch

        # make a vector of random integers between 0 and samples-1
        indexes = random.sample(range(self.samples), self.batch_size)
        hdf5_file = tables.open_file("frogtrainingdata_noambiguities.hdf5", mode="r")
        E_real_batch = hdf5_file.root.E_real[indexes, :]
        E_imag_batch = hdf5_file.root.E_imag[indexes, :]
        E_appended_batch = np.append(E_real_batch, E_imag_batch, 1)
        frog_batch = hdf5_file.root.frog[indexes, :]
        hdf5_file.close()

        self.batch_index += self.batch_size

        return  frog_batch, E_appended_batch







    def evaluate_on_test_data(self, samples):

        # this is used to evaluate the mean squared error of the data after every epoch

        hdf5_file = tables.open_file("frogtestdata_noambiguities.hdf5", mode="r")
        E_real_eval = hdf5_file.root.E_real[:samples, :]
        E_imag_eval = hdf5_file.root.E_imag[:samples, :]
        E_appended_eval = np.append(E_real_eval, E_imag_eval, 1)
        frog_eval = hdf5_file.root.frog[:samples, :]
        hdf5_file.close()

        return frog_eval, E_appended_eval

    def evaluate_on_train_data(self, samples):

        # this is used to evaluate the mean squared error of the data after every epoch

        hdf5_file = tables.open_file("frogtrainingdata_noambiguities.hdf5", mode="r")
        E_real_eval = hdf5_file.root.E_real[:samples, :]
        E_imag_eval = hdf5_file.root.E_imag[:samples, :]
        E_appended_eval = np.append(E_real_eval, E_imag_eval, 1)
        frog_eval = hdf5_file.root.frog[:samples, :]
        hdf5_file.close()

        return frog_eval, E_appended_eval


def plot_predictions(x_in, y_in, axis, fig, set, modelname, epoch):

    mses = []
    predictions = sess.run(y_pred, feed_dict={x: x_in,
                                              y_true: y_in})

    for ax, index in zip([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]):

        mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1),
                                        y_true: y_in[index].reshape(1, -1)})
        mses.append(mse)

        # plot E(t) retrieved
        axis[0][ax].cla()
        axis[0][ax].plot(t, predictions[index, :64], color="blue")
        axis[0][ax].plot(t, predictions[index, 64:], color="red")

        axis[0][ax].text(0.1, 1.3, "Epoch : {}".format(i + 1),
                        transform=axis[0][ax].transAxes, backgroundcolor='white')
        axis[0][ax].text(0.1, 1.15, "MSE: " + str(mse),
                         transform=axis[0][ax].transAxes, backgroundcolor='white')
        axis[0][ax].text(0.1, 1, "prediction ["+set+" set]", transform=axis[0][ax].transAxes,
                         backgroundcolor='white')


        # plot retieved FROG
        E_pred = np.array(predictions[index, :64]) + 1j * np.array(predictions[index, 64:])
        frogtrace, _, _ = plot_frog(E=E_pred, t=t, w=w, dt=dt, w0=w0, plot=False)
        axis[1][ax].pcolormesh(frogtrace, cmap="jet")
        axis[1][ax].text(0.1, 1, "reconstructed frog trace", transform=axis[1][ax].transAxes,
                        backgroundcolor='white')

        # plot E(t) retrieved
        axis[2][ax].cla()
        axis[2][ax].plot(t, y_in[index, :64], color="blue")
        axis[2][ax].plot(t, y_in[index, 64:], color="red")
        axis[2][ax].text(0.1, 1, "actual ["+set+" set]", transform=axis[2][ax].transAxes,
                         backgroundcolor='white')

        # plot  actual FROG
        axis[3][ax].pcolormesh(x_in[index].reshape(64, 64), cmap='jet')
        axis[3][ax].text(0.1, 1, "actual frog trace", transform=axis[3][ax].transAxes,
                        backgroundcolor='white')

        # set all x labels to none
        for index in range(4):
            for index2 in range(6):
                axis[index][index2].set_xticks([])
                axis[index][index2].set_yticks([])

    print("mses: ", mses)
    print("avg : ", (1/len(mses))*np.sum(np.array(mses)))

    # save image
    dir = "/home/zom/PythonProjects/frog_simulation_machinelearning/pictures/"+modelname+"/"+set+"/"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig.savefig(dir+str(epoch)+".png")


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')


# convolutional layer
def convolutional_layer(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b


# dense layer
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# placeholders
x = tf.placeholder(tf.float32, shape=[None, int(64*64)])
y_true = tf.placeholder(tf.float32, shape=[None, 128])

# layers
x_image = tf.reshape(x, [-1, 64, 64, 1])
# shape = [sizex, sizey, channels, filters/features]
convo_1 = convolutional_layer(x_image, shape=[4, 4, 1, 32], activate='none', stride=[2, 2])
convo_2 = convolutional_layer(convo_1, shape=[2, 2, 32, 32], activate='none', stride=[2, 2])
convo_3 = convolutional_layer(convo_2, shape=[1, 1, 32, 32], activate='leaky', stride=[1, 1])

#convo_3_flat = tf.reshape(convo_3, [-1, 58*106*32])
convo_3_flat = tf.contrib.layers.flatten(convo_3)
#full_layer_one = tf.nn.relu(normal_full_layer(convo_3_flat, 512))
full_layer_one = normal_full_layer(convo_3_flat, 512)

#dropout
hold_prob = tf.constant(0.5, dtype=tf.float32)
dropout_layer = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(dropout_layer, 128)
#y_pred = normal_full_layer(full_layer_one, 128)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# initialize data object
get_data = GetData(batch_size=500)

test_mse_tb = tf.summary.scalar("test_mse", loss)
train_mse_tb = tf.summary.scalar("train_mse", loss)
_, t, w, dt, w0, _ = retrieve_data(plot_frog_bool=False, print_size=False)
saver = tf.train.Saver()

#epochs = 500
epochs = 5000
print("5000 epoch")

if __name__ == "__main__":
    modelname = "60k_samples_leaky_dropout_notrandindexes_batchsize500"

    # create figures to visualize predictions in realtime
    fig1, ax1 = plt.subplots(4, 6, figsize=(14, 8))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                        wspace=0.1, hspace=0.1)

    fig2, ax2 = plt.subplots(4, 6, figsize=(14, 8))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                        wspace=0.1, hspace=0.1)

    plt.ion()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./tensorboard_graph/"+modelname)
        # summaries = tf.summary.merge_all()

        for i in range(epochs):
            print("Epoch : {}".format(i+1))

            # iterate through every sample in the training set
            dots = 0
            while get_data.batch_index < get_data.samples:

                percent = 50 * get_data.batch_index / get_data.samples
                if percent - dots > 1:
                    print(".", end="", flush=True)
                    dots += 1

                batch_x, batch_y = get_data.next_batch()
                # retrieve random samples
                #batch_x, batch_y = get_data.next_batch_random()
                #print('batch retrieved')
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

            print("")

            # view the mean squared error of the test data
            batch_x_test, batch_y_test = get_data.evaluate_on_test_data(samples=500)
            print("test MSE: ", sess.run(loss, feed_dict={x: batch_x_test, y_true: batch_y_test}), "\n")
            summ = sess.run(test_mse_tb, feed_dict={x: batch_x_test, y_true: batch_y_test})
            writer.add_summary(summ, global_step=i+1)

            # view the mean squared error of the train data
            batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=500)
            print("train MSE: ", sess.run(loss, feed_dict={x: batch_x_train, y_true: batch_y_train}))
            summ = sess.run(train_mse_tb, feed_dict={x: batch_x_train, y_true: batch_y_train})
            writer.add_summary(summ, global_step=i+1)

            writer.flush()

            # every x steps plot predictions
            if (i+1) % 100 == 0 or (i+1) <=15:

                # update the plot

                batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=500)
                plot_predictions(x_in=batch_x_train, y_in=batch_y_train, axis=ax1, fig=fig1,
                                 set="train", modelname=modelname, epoch=i+1)

                batch_x_test, batch_y_test = get_data.evaluate_on_test_data(samples=500)
                plot_predictions(x_in=batch_x_test, y_in=batch_y_test, axis=ax2, fig=fig2,
                                 set="test", modelname=modelname, epoch=i+1)

                plt.show()
                plt.pause(0.001)


            # return the index to 0
            get_data.batch_index = 0

        saver.save(sess, "models/"+modelname+".ckpt")

    plt.ioff()
    plt.show()
