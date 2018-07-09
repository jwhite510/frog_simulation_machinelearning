from tensorflow.contrib.keras import models
import matplotlib.pyplot as plt
from generate_data import retrieve_data
import numpy as np

# model = models.load_model('./2000_epoch_1000_sample/2000_epochs_1000_samples.hdf5')

model = models.load_model('./1000_epoch_1000_sample/1000_epoch_1000_sample.hdf5')


tests = 3
fig, ax = plt.subplots(3, tests)
for j in range(tests):
    # generate data
    E_actual, t, frogtrace_flat = retrieve_data(plot_frog_bool=False, print_size=False)

    # pred = model.predict(frog[index_test].reshape(1, 58, 106, 1))
    pred = model.predict(frogtrace_flat.reshape(1, 58, 106, 1))

    E_imag_pred = pred[0][128:]
    E_real_pred = pred[0][:128]
    E_pred = E_real_pred + 1j * E_imag_pred

    ax[0][j].plot(t, np.abs(E_actual), color='black')
    ax[0][j].set_xticks([])
    ax[0][j].set_yticks([])
    axtwin = ax[0][j].twinx()

    if j == tests-1:
        axtwin.set_ylabel('$\phi (t)$', color='green')

    # axtwin.set_yticks([])
    axtwin.plot(t, np.unwrap(np.angle(E_actual)), color='green'
                                                        '')
    ## for plotting imaginary and real part:
    # ax[0][j].plot(t, np.abs(E_actual), color='black', linestyle='dashed', alpha=0.5)
    # ax[0][j].plot(t, np.real(E_actual), color='blue')
    # ax[0][j].plot(t, np.imag(E_actual), color='red')

    ax[1][j].pcolormesh(frogtrace_flat.reshape(58, 106), cmap='jet')
    ax[1][j].set_xticks([])
    ax[1][j].set_yticks([])

    ax[2][j].plot(t, np.abs(E_pred), color='black')
    ax[2][j].set_xticks([])
    ax[2][j].set_yticks([])
    axtwin = ax[2][j].twinx()

    if j == tests-1:
        axtwin.set_ylabel('$\phi (t)$', color='green')
    # axtwin.set_yticks([])

    axtwin.plot(t, np.unwrap(np.angle(E_pred)), color='green')

    ## for plotting imaginary and real part:
    # ax[2][j].plot(t, np.abs(E_pred), color='black', linestyle='dashed', alpha=0.5)
    # ax[2][j].plot(t, np.real(E_pred), color='blue')
    # ax[2][j].plot(t, np.imag(E_pred), color='red')


    if j == 0:
        ax[0][j].set_ylabel('Actual E(t)')
        ax[1][j].set_ylabel('FROG trace')
        ax[2][j].set_ylabel('Retrieved E(t)')






plt.show()