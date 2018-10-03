import tensorflow as tf
import tables
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof
from generate_data import retrieve_data

from network_tf import *

# _, ax1 = plt.subplots(1, 3)
# _, ax2 = plt.subplots(1, 3)
# _, ax3 = plt.subplots(1, 3)
# _, ax4 = plt.subplots(1, 3)
# _, ax5 = plt.subplots(1, 3)
# _, ax6 = plt.subplots(1, 3)


def plot_results(batch_x_train, batch_y_train, predictions):

    fig = plt.figure(figsize=(7,7))
    gs = fig.add_gridspec(3, 3)


    for index, gs_col_index in zip([7, 2, 3], [0, 1, 2]):

        axis = fig.add_subplot(gs[0, gs_col_index])
        axis.pcolormesh(batch_x_train[index].reshape(64, 64), cmap='jet')
        axis.set_xticks([])
        axis.set_yticks([])
        if gs_col_index == 0:
            axis.set_ylabel('Input FROG trace')

        axis = fig.add_subplot(gs[1, gs_col_index])
        axis.plot(predictions[index, :64], color='blue')
        axis.plot(predictions[index, 64:], color='red')
        complex_proj = np.array(predictions[index, :64]) + 1j * np.array(predictions[index, 64:])
        phase = np.unwrap(np.angle(complex_proj))
        axtwin = axis.twinx()
        axtwin.plot(np.array(range(64))[10:54], phase[10:54], color='green', alpha=0.7, label='phase', linestyle='dashed')
        if gs_col_index == 0:
            axis.set_ylabel('Predicted $E(t)$')
        axis.set_ylim([-0.5, 1])
        axis.set_xticks([])
        axis.set_yticks([])
        axtwin.set_yticks([])

        axis = fig.add_subplot(gs[2,gs_col_index])
        reale, = axis.plot(batch_y_train[index, :64], color='blue', label='real $E(t)$')
        image, = axis.plot(batch_y_train[index, 64:], color='red', label='imag $E(t)$')
        complex_proj = np.array(batch_y_train[index, :64]) + 1j * np.array(batch_y_train[index, 64:])
        phase = np.unwrap(np.angle(complex_proj))
        axtwin = axis.twinx()
        axtwin.plot(np.array(range(64))[10:54], phase[10:54], color='green', alpha=0.7, label='phase', linestyle='dashed')
        axis.set_ylim([-0.5, 1])
        axis.set_xticks([])
        axis.set_yticks([])
        axtwin.set_yticks([])

        if gs_col_index == 0:
            axis.set_ylabel('Actual $E(t)$')
            xspacing = np.linspace(0.5, 1.7, 3)
            l1 = axis.legend([reale], ['real $E(t)$'], bbox_to_anchor=(xspacing[0], -0.02))
            l2 = axis.legend([image], ['imag $E(t)$'], bbox_to_anchor=(xspacing[1], -0.02))
            axis.add_artist(l1)
            axtwin.legend(bbox_to_anchor=(xspacing[2]-0.1, -0.02))

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0, hspace=0, top=0.95, bottom=0.07)
    plt.savefig('./resultsplotted.png')




# restore
with tf.Session() as sess:

    saver.restore(sess, "models/120k_samples_holdout_01_amp5.ckpt")

    batch_x_eval, batch_y_eval = get_data.evaluate_on_test_data(samples=10)

    print("test MSE: ", sess.run(loss, feed_dict={x: batch_x_eval, y_true: batch_y_eval}))

    #predictions = sess.run(y_pred, feed_dict={x: batch_x_eval})


    # PLOT PREDICTIONS
    mses = []
    batch_x_train, batch_y_train = get_data.evaluate_on_test_data(samples=10)
    predictions = sess.run(y_pred, feed_dict={x: batch_x_train, y_true: batch_y_train})
    # for ax, index in zip([ax1, ax2, ax3, ax4, ax5, ax6], [0, 1, 2, 3, 4, 5]):
    #     mse = sess.run(loss,
    #                    feed_dict={x: batch_x_train[index].reshape(1, -1), y_true: batch_y_train[index].reshape(1, -1)})
    #     mses.append(mse)
    #     ax[0].cla()
    #     ax[0].plot(t, predictions[index, :64], color="blue")
    #     ax[0].plot(t, predictions[index, 64:], color="red")
    #     ax[0].set_title("prediction [test set]")
    #     ax[0].text(0.5, 0.5, "MSE: " + str(mse), transform=ax[0].transAxes, backgroundcolor='white')
    #     ax[1].cla()
    #     ax[1].plot(t, batch_y_train[index, :64], color="blue")
    #     ax[1].plot(t, batch_y_train[index, 64:], color="red")
    #     ax[1].set_title("actual [train set]")
    #     ax[2].pcolormesh(batch_x_train[index].reshape(64,64), cmap='jet')
    plot_results(batch_x_train, batch_y_train, predictions)
    plt.show()
    print("mses: ", mses)
    print("avg : ", (1 / len(mses)) * np.sum(np.array(mses)))

