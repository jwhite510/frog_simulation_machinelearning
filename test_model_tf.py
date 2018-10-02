import tensorflow as tf
import tables
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof
from generate_data import retrieve_data

from network_tf import *

_, ax1 = plt.subplots(1, 2)
_, ax2 = plt.subplots(1, 2)
_, ax3 = plt.subplots(1, 2)
_, ax4 = plt.subplots(1, 2)
_, ax5 = plt.subplots(1, 2)
_, ax6 = plt.subplots(1, 2)

# restore
with tf.Session() as sess:

    saver.restore(sess, "models/paper_10_samples.ckpt")

    batch_x_eval, batch_y_eval = get_data.evaluate_on_test_data()

    print("test MSE: ", sess.run(loss, feed_dict={x: batch_x_eval, y_true: batch_y_eval}))

    #predictions = sess.run(y_pred, feed_dict={x: batch_x_eval})


    # PLOT PREDICTIONS
    mses = []
    batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=10)
    predictions = sess.run(y_pred, feed_dict={x: batch_x_train, y_true: batch_y_train})
    for ax, index in zip([ax1, ax2, ax3, ax4, ax5, ax6], [0, 1, 2, 3, 4, 5]):
        mse = sess.run(loss,
                       feed_dict={x: batch_x_train[index].reshape(1, -1), y_true: batch_y_train[index].reshape(1, -1)})
        mses.append(mse)
        ax[0].cla()
        ax[0].plot(t, predictions[index, :64], color="blue")
        ax[0].plot(t, predictions[index, 64:], color="red")
        ax[0].set_title("prediction [train set]")
        ax[0].text(0.5, 0.5, "MSE: " + str(mse), transform=ax[0].transAxes, backgroundcolor='white')
        ax[1].cla()
        ax[1].plot(t, batch_y_train[index, :64], color="blue")
        ax[1].plot(t, batch_y_train[index, 64:], color="red")
        ax[1].set_title("actual [train set]")
    plt.show()
    print("mses: ", mses)
    print("avg : ", (1 / len(mses)) * np.sum(np.array(mses)))

