from tensorflow.contrib.keras import models
import matplotlib.pyplot as plt
from generate_data import retrieve_data
from frognet1 import plot_frog
import numpy as np
import tables



def test_data_show():
    for j in range(tests):

        # generate data
        _, t, w, dt, w0, _ = retrieve_data(plot_frog_bool=False, print_size=False)

        E_actual = E_real[j] + 1j * E_imag[j]

        frogtrace_flat = frog[j]

        # generate model prediction
        pred = model.predict(frogtrace_flat.reshape(1, 58, 106, 1))

        E_imag_pred = pred[0][128:]
        E_real_pred = pred[0][:128]
        E_pred = E_real_pred + 1j * E_imag_pred


        # plot actual E(t) and phi(t)
        ax1 = plt.subplot2grid((4,2), (0,0), rowspan=1, colspan=1)
        ax1.plot(t, np.real(E_actual), color="blue", alpha=0.5)
        ax1.plot(t, np.imag(E_actual), color="red", alpha=0.5)
        ax1.plot(t, np.abs(E_actual), color='black')
        ax1.set_title("Actual $E(t)$")
        ax1.set_xticks([])
        ax1.set_yticks([])
        axtwin = ax1.twinx()
        if j == tests-1:
            axtwin.set_ylabel('$\phi (t)$', color='green')
        axtwin.plot(t, np.unwrap(np.angle(E_actual)), color='green')

        # plot actual E(w) and phi(w)
        E_w_actual = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_actual)))
        ax1_2 = plt.subplot2grid((4,2), (0,1), rowspan=1, colspan=1)
        ax1_2.plot(w, np.real(E_w_actual), color="blue", alpha=0.5)
        ax1_2.plot(w, np.imag(E_w_actual), color="red", alpha=0.5)
        ax1_2.plot(w, np.abs(E_w_actual), color='black')
        ax1_2.set_title("Actual $E(\omega)$")
        ax1_2.set_xticks([])
        ax1_2.set_yticks([])
        axtwin = ax1_2.twinx()
        if j == tests-1:
            axtwin.set_ylabel('$\phi (\omega)$', color='green')
        axtwin.plot(w, np.unwrap(np.angle(E_w_actual)), color='green')

        # plot actual FROG trace
        ax2 = plt.subplot2grid((4, 2), (1, 0), rowspan=1, colspan=2)
        ax2.pcolormesh(frogtrace_flat.reshape(58, 106), cmap='jet')
        ax2.set_title("Network Input FROG trace")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # plot retrieved E(t) and phi(t)
        ax3 = plt.subplot2grid((4, 2), (2, 0), rowspan=1, colspan=1)
        ax3.plot(t, np.real(E_pred), color="blue", alpha=0.5)
        ax3.plot(t, np.imag(E_pred), color="red", alpha=0.5)
        ax3.plot(t, np.abs(E_pred), color='black')
        ax3.set_title("Retrieved $E(t)$")
        ax3.set_xticks([])
        ax3.set_yticks([])
        axtwin = ax3.twinx()
        if j == tests-1:
            axtwin.set_ylabel('$\phi (t)$', color='green')
        axtwin.plot(t, np.unwrap(np.angle(E_pred)), color='green')

        # plot retrieved E(w) and phi(w)
        E_w_pred = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_pred)))
        ax3_2 = plt.subplot2grid((4,2), (2, 1), rowspan=1, colspan=1)
        ax3_2.plot(w, np.real(E_w_pred), color="blue", alpha=0.5)
        ax3_2.plot(w, np.imag(E_w_pred), color="red", alpha=0.5)
        ax3_2.plot(w, np.abs(E_w_pred), color="black")
        ax3_2.set_title("Retrieved $E(\omega)$")
        ax3_2.set_xticks([])
        ax3_2.set_yticks([])
        axtwin = ax3_2.twinx()
        if j == tests-1:
            axtwin.set_ylabel('$\phi (\omega)$', color='green')
        axtwin.plot(w, np.unwrap(np.angle(E_w_pred)), color='green')


        # plot the reconstructed FROG trace
        frogtrace, tau_frog, w_frog = plot_frog(E=E_pred, t=t, w=w, dt=dt, w0=w0, plot=False)
        ax4 = plt.subplot2grid((4, 2), (3, 0), rowspan=1, colspan=2)
        ax4.pcolormesh(frogtrace, cmap="jet")
        ax4.set_title("Network Reconstructed FROG trace")
        ax4.set_xticks([])
        ax4.set_yticks([])

        # add y labels on left side of graph only
        if j == 0:
            ax1.set_ylabel('Actual E(t)')
            ax2.set_ylabel('FROG trace')
            ax3.set_ylabel('Retrieved E(t)')
            ax4.set_ylabel("Reconstructed FROG trace")

# load model
model = models.load_model("./model.hdf5")

# load test data
hdf5_file = tables.open_file('frogtestdata.hdf5', mode='r')
E_real = hdf5_file.root.E_real[:, :]
E_imag = hdf5_file.root.E_imag[:, :]
frog = hdf5_file.root.frog[:, :]
# print((frog[1].reshape(58, 106)))
# image = frog[1].reshape(58, 106)
# plt.imshow(image)
# plt.show()
E_apended = np.concatenate((E_real, E_imag), 1)
hdf5_file.close()


results = model.evaluate(frog.reshape(-1, 58, 106, 1), E_apended)
print("mse: ", results)

tests = 1
fig = plt.figure(figsize=(9, 10))
test_data_show()
plt.show()

