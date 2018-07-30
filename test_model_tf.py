import tensorflow as tf
import tables
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof
from generate_data import retrieve_data

from network_tf import *


# restore
with tf.Session() as sess:

    saver.restore(sess, "models/model.ckpt")

    batch_x_eval, batch_y_eval = get_data.evaluate_on_test_data()

    print("MSE: ", sess.run(loss, feed_dict={x: batch_x_eval, y_true: batch_y_eval}))

    predictions = sess.run(y_pred, feed_dict={x: batch_x_eval})

