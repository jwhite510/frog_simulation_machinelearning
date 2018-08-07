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
    ax[1].pcolormesh(frog.reshape(64, 64), cmap='jet')

_, t, w, dt, w0, _ = retrieve_data(plot_frog_bool=False, print_size=False)



def find_nearly_matching_frog_traces(compare_index, msemax, search_size, file):

    hdf5_file = tables.open_file(filename=file, mode="r")
    E_real = hdf5_file.root.E_real[:, :]
    E_imag = hdf5_file.root.E_imag[:, :]
    frog = hdf5_file.root.frog[:, :]
    hdf5_file.close()

    compare_frog = frog[compare_index, :]
    compare_E = np.array(E_real[compare_index, :]) + 1j * np.array(E_imag[compare_index, :])

    similar_frog_traces = []
    their_E = []
    their_index = []
    their_mse = []

    for i in range(search_size):
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


def find_nearly_matching_E_field(compare_index, msemax, search_size, file):
    hdf5_file = tables.open_file(filename=file, mode="r")
    E_real = hdf5_file.root.E_real[:, :]
    E_imag = hdf5_file.root.E_imag[:, :]
    frog = hdf5_file.root.frog[:, :]
    hdf5_file.close()

    compare_frog = frog[compare_index, :]
    compare_E = np.array(E_real[compare_index, :]) + 1j * np.array(E_imag[compare_index, :])
    compare_E_appended = np.append(np.array(E_real[compare_index, :]), np.array(E_imag[compare_index, :]))

    similar_E_fields = []
    their_frog = []
    their_index = []
    their_mse = []

    for i in range(search_size):
        # calc mse
        E_test = np.append(np.array(E_real[i]), np.array(E_imag[i]))

        mse = (1/len(compare_E_appended)) * np.sum((E_test - compare_E_appended)**2)

        if mse < msemax:
            similar_E_fields.append(np.array(E_real[i]) + 1j * np.array(E_imag[i]))
            their_frog.append(frog[i])
            their_index.append(i)
            their_mse.append(mse)

        if i % 100 == 0:
            print('step : ', i)
            print('number of similar:', len(similar_E_fields))



    plot_frog_and_E(frog=compare_frog, E=compare_E, title='compare E, index: {}'.format(str(compare_index)))


    for frog, E, index1 in zip(their_frog, similar_E_fields, their_index):

        plot_frog_and_E(frog=np.array(frog), E=E,
                        title='mse<{}, index: {}'.format(str(msemax), str(index1)))

    plt.show()





# find_nearly_matching_frog_traces(compare_index=0, msemax=0.01, search_size=1000,
#                                  file="frogtrainingdata_noambiguities.hdf5")

find_nearly_matching_E_field(compare_index=1, msemax=0.0001, search_size=10000,
                                 file="frogtrainingdata_noambiguities.hdf5")