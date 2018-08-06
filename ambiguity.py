import tables
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof
from generate_data import retrieve_data
import time
from frognet1 import plot_frog
import os


def plot_frog_and_E(frog, E, title):
    # plot the comparing trace and field
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title(title)
    ax[0].plot(t, np.real(E), color='blue')
    ax[0].plot(t, np.imag(E), color='red')
    ax[0].plot(t, np.abs(E), color='black', linestyle='dashed')
    ax[1].pcolormesh(frog.reshape(64, 64))

_, t, w, dt, w0, _ = retrieve_data(plot_frog_bool=False, print_size=False)

hdf5_file = tables.open_file("frogtrainingdata_noambiguities.hdf5", mode="r")
E_real = hdf5_file.root.E_real[:, :]
E_imag = hdf5_file.root.E_imag[:, :]
frog = hdf5_file.root.frog[:, :]
hdf5_file.close()


compare_index = 18
compare_frog = frog[compare_index, :]
compare_E = np.array(E_real[compare_index, :]) + 1j * np.array(E_imag[compare_index, :])



similar_frog_traces = []
their_E = []
their_index = []
their_mse = []

msemax = 0.05

for i in range(100):

    # calc mse
    mse = (1 / (len(frog[i]))) * np.sum((frog[i] - compare_frog)**2)

    if mse < msemax:
        similar_frog_traces.append(frog[i])
        their_E.append(np.array(E_real[i]) + 1j * np.array(E_imag[i]))
        their_index.append(i)
        their_mse.append(mse)

    if i % 100 == 0:
        print('step : ', i)
        print('number of similar:', len(similar_frog_traces))


plot_frog_and_E(frog=compare_frog, E=compare_E, title='compare E, index: {}'.format(str(compare_index)))


for frog, E, index1 in zip(similar_frog_traces, their_E, their_index):

    plot_frog_and_E(frog=np.array(frog), E=E,
                    title='mse<{}, index: {}'.format(str(msemax), str(index1)))

plt.show()