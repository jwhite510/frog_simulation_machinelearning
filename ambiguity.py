import tables
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof
from generate_data import retrieve_data
import time
from frognet1 import plot_frog
import os


def plot_frog_and_E(frog, E, title, mse, integrate):
    # plot the comparing trace and field
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    t_0_index = np.argmin(np.abs(t - 0.0))

    # plot horizontal and vertical lines
    ax[0].plot([0, 0], [-1, 1], color='black', alpha=0.5)
    ax[0].plot([t[0], t[-1]], [0, 0], color='black', alpha=0.5)

    # plot real, imag and abs E
    ax[0].plot(t, np.real(E), color='blue', label='real $E(t)$', linewidth=2)
    ax[0].plot(t, np.imag(E), color='red', label='imag $E(t)$', linewidth=2)
    ax[0].plot(t, np.abs(E), color='black', linestyle='dashed', label='abs $E(t)$')

    # add details
    ax[0].legend(loc=2)
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('$E(t)$')

    # plot frog trace
    ax[1].pcolormesh(frog.reshape(64, 64), cmap='jet')

    if mse:
        mse = round(mse, 5)
    ax[0].text(0.9, 1.05, 'compared FROG trace mse: {}'.format(str(mse)), transform=ax[0].transAxes,
               backgroundcolor='white')

    axtwin = ax[0].twinx()
    axtwin.text(0.05, 0.02, title, transform=ax[0].transAxes, backgroundcolor='white')
    axtwin.plot(t, np.unwrap(np.angle(E)), color='green', label='$\phi (t)$', linestyle='dashed')
    axtwin.legend(loc=1)

    if integrate=='real':

        ax[0].fill_between(t[:t_0_index], 0, np.real(E)[:t_0_index], color='cyan')
        ax[0].fill_between(t[t_0_index + 1:], 0, np.real(E)[t_0_index + 1:], color='lightgreen')

        integral_left_side = dt * np.sum(np.real(E[:t_0_index]))
        integral_right_side = dt * np.sum(np.real(E[t_0_index+1:]))

        integral_right_side = round(integral_right_side, 18)
        integral_left_side = round(integral_left_side, 18)

        axtwin.text(-0.3, 1.1, 'real integral left side: {}'.format(integral_left_side),
                    transform=axtwin.transAxes, backgroundcolor='cyan')
        axtwin.text(-0.3, 1.02, 'real integral right side: {}'.format(integral_right_side),
                    transform=axtwin.transAxes, backgroundcolor='lightgreen')

    elif integrate=='abs':

        ax[0].fill_between(t[:t_0_index], 0, np.abs(E)[:t_0_index], color='cyan')
        ax[0].fill_between(t[t_0_index + 1:], 0, np.abs(E)[t_0_index + 1:], color='lightgreen')

        integral_left_side = dt * np.sum(np.abs(E[:t_0_index]))
        integral_right_side = dt * np.sum(np.abs(E[t_0_index + 1:]))

        integral_right_side = round(integral_right_side, 18)
        integral_left_side = round(integral_left_side, 18)

        axtwin.text(-0.3, 1.1, 'abs integral left side: {}'.format(integral_left_side),
                    transform=axtwin.transAxes, backgroundcolor='cyan')
        axtwin.text(-0.3, 1.02, 'abs integral right side: {}'.format(integral_right_side),
                    transform=axtwin.transAxes, backgroundcolor='lightgreen')



def find_nearly_matching_frog_traces(compare_index, msemax, search_size, file, integrate):

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


    plot_frog_and_E(frog=compare_frog, E=compare_E, title='compare FROG, index: {}'.format(str(compare_index)),
                    mse=None, integrate=integrate)


    for frog, E, index1, mse1 in zip(similar_frog_traces, their_E, their_index, their_mse):

        plot_frog_and_E(frog=np.array(frog), E=E,
                        title='mse<{}, index: {}'.format(str(msemax), str(index1)), mse=mse1,
                        integrate=integrate)

    plt.show()


def find_nearly_matching_E_field(compare_index, msemax, search_size, file, integrate):
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



    plot_frog_and_E(frog=compare_frog, E=compare_E, title='compare E, index: {}'.format(str(compare_index)),
                    mse=None, integrate=integrate)


    for frog, E, index1, mse1 in zip(their_frog, similar_E_fields, their_index, their_mse):

        plot_frog_and_E(frog=np.array(frog), E=E,
                        title='mse<{}, index: {}'.format(str(msemax), str(index1)), mse=mse1,
                        integrate=integrate)

    plt.show()



_, t, w, dt, w0, _ = retrieve_data(plot_frog_bool=False, print_size=False)

# find_nearly_matching_frog_traces(compare_index=4, msemax=0.006, search_size=10000,
#                                  file="frogtrainingdata.hdf5", integrate='real')

find_nearly_matching_frog_traces(compare_index=3, msemax=0.01, search_size=14000,
                                 file="frogtrainingdata.hdf5", integrate='real')


# find_nearly_matching_E_field(compare_index=1, msemax=0.0001, search_size=10000,
#                                  file="frogtrainingdata.hdf5")