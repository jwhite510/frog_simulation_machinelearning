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

    def evaluate_on_test_data(self):

        # this is used to evaluate the mean squared error of the data after every epoch

        hdf5_file = tables.open_file("frogtestdata.hdf5", mode="r")
        E_real_eval = hdf5_file.root.E_real[:, :]
        E_imag_eval = hdf5_file.root.E_imag[:, :]
        E_appended_eval = np.append(E_real_eval, E_imag_eval, 1)
        frog_eval = hdf5_file.root.frog[:, :]
        hdf5_file.close()

        return frog_eval, E_appended_eval

    def evaluate_on_train_data(self):

        # this is used to evaluate the mean squared error of the data after every epoch

        hdf5_file = tables.open_file("frogtrainingdata.hdf5", mode="r")
        E_real_eval = hdf5_file.root.E_real[:500, :]
        E_imag_eval = hdf5_file.root.E_imag[:500, :]
        E_appended_eval = np.append(E_real_eval, E_imag_eval, 1)
        frog_eval = hdf5_file.root.frog[:500, :]
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
convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 8])
convo_2 = convolutional_layer(convo_1, shape=[5, 5, 8, 8])
convo_3 = convolutional_layer(convo_2, shape=[5, 5, 8, 8])

print("8 8 8")
#convo_2_flat = tf.reshape(convo_2, [-1, 58*106*9])
convo_3_flat = tf.reshape(convo_3, [-1, 58*106*8])
print("512")
full_layer_one = tf.nn.relu(normal_full_layer(convo_3_flat, 512))
full_layer_two = tf.nn.relu(normal_full_layer(full_layer_one, 512))
y_pred = normal_full_layer(full_layer_two, 256)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# initialize data object
get_data = GetData(batch_size=500)

test_mse_tb = tf.summary.scalar("test_mse", loss)
train_mse_tb = tf.summary.scalar("train_mse", loss)
_, t, _, _, _, _ = retrieve_data(False, False)
saver = tf.train.Saver()

epochs = 300

if __name__ == "__main__":
    modelname = "first_test"
    # create figures to visualize predictions in realtime
    _, ax1 = plt.subplots(1, 2)
    _, ax2 = plt.subplots(1, 2)
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
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

            print("")

            # view the mean squared error of the train data
            batch_x_train, batch_y_train = get_data.evaluate_on_train_data()
            print("train MSE: ", sess.run(loss, feed_dict={x: batch_x_train, y_true: batch_y_train}))

            # view the mean squared error of the test data
            batch_x_test, batch_y_test = get_data.evaluate_on_test_data()
            print("test MSE: ", sess.run(loss, feed_dict={x: batch_x_test, y_true: batch_y_test}), "\n")

            predictions = sess.run(y_pred, feed_dict={x: batch_x_test, y_true: batch_y_test})

            # add summaries for tensorboard
            summ = sess.run(test_mse_tb, feed_dict={x: batch_x_test, y_true: batch_y_test})
            writer.add_summary(summ, global_step=i+1)

            summ = sess.run(train_mse_tb, feed_dict={x: batch_x_train, y_true: batch_y_train})
            writer.add_summary(summ, global_step=i+1)

            writer.flush()

            # update the plot
            for ax, index in zip([ax1, ax2], [1, 2]):

                ax[0].cla()
                ax[0].plot(t, predictions[index, :128], color="blue")
                ax[0].plot(t, predictions[index, 128:], color="red")
                ax[0].set_title("prediction [train set]")

                ax[1].cla()
                ax[1].plot(t, batch_y_test[index, :128], color="blue")
                ax[1].plot(t, batch_y_test[index, 128:], color="red")
                ax[1].set_title("actual [train set]")

            plt.show()
            plt.pause(0.001)

            # return the index to 0
            get_data.batch_index = 0

        saver.save(sess, "models/"+modelname+".ckpt")
