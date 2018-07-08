from tensorflow.contrib.keras import models
import matplotlib.pyplot as plt
from generate_data import retrieve_data
import numpy as np
import tables

model = models.load_model('model.hdf5')

_, t, _ = retrieve_data(False, False)

def test_sample(index):

    index_test = index

    E_actual = E_real[index_test] + 1j * E_imag[index_test]

    pred = model.predict(frog[index_test].reshape(1, 58, 106, 1))

    E_imag_pred = pred[0][128:]
    E_real_pred = pred[0][:128]
    E_pred = E_real_pred + 1j * E_imag_pred

    fig, ax = plt.subplots(2, 2)
    # ax[0][0].pcolormesh(frog[index_test].reshape(58, 106), cmap='jet')
    ax[1][1].plot(t, np.abs(E_pred), color='blue', linestyle='dashed')
    ax[1][1].plot(t, np.real(E_pred), color='blue')
    ax[1][1].plot(t, np.imag(E_pred), color='red')


    ax[0][0].pcolormesh(frog[index_test].reshape(58, 106), cmap='jet')
    ax[0][1].plot(t, np.abs(E_actual), color='blue', linestyle='dashed')
    ax[0][1].plot(t, np.real(E_actual), color='blue')
    ax[0][1].plot(t, np.imag(E_actual), color='red')



hdf5_file = tables.open_file('frogtrainingdata.hdf5', mode='r')

index = 99
E_real = hdf5_file.root.E_real[:, :]
E_imag = hdf5_file.root.E_imag[:, :]
frog = hdf5_file.root.frog[:, :]

E_apended = np.concatenate((E_real, E_imag), 1)

hdf5_file.close()





test_sample(0)
test_sample(1)
test_sample(2)
test_sample(3)
test_sample(5)
test_sample(6)
test_sample(7)
test_sample(8)

plt.show()