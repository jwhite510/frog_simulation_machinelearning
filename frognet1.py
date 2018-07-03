import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy import interpolate


def generateE(plot, GVD, TOD):

    N = 2000
    dt = 0.1e-15
    df = 1 / (dt * N)

    # time axis
    t = dt * np.linspace(-N/2, N/2, N)

    # frequency axis
    w = 2 * np.pi * df * np.linspace(-N/2, N/2, N)

    # define pulse in time
    tau = 5e-15
    w0 = 4e15 # rad/s

    E_t = np.exp(1j * w0 * t) * np.exp(-t**2/tau**2)

    # # ADDING MULTIPLE PULSES
    # E_t = E_t + np.exp(1j * w0 * t) * np.exp(-(t + 15e-15) ** 2 / tau ** 2)
    # E_t = E_t + np.exp(1j * w0 * t) * np.exp(-(t - 15e-15) ** 2 / tau ** 2)
    # ##########

    E_w = np.fft.fftshift(np.fft.fft(E_t))

    # GVD = 1e-30
    k = GVD * (w - w0)**2 + TOD * (w - w0)**3

    # get zero phase
    E_w_zero_phase = np.unwrap(np.angle(E_w))

    # apply transfer function
    z = 1  # distance
    E_w_prop = E_w * np.exp(1j * k * z)

    # calculate spectral phase
    E_w_prop_phase = np.unwrap(np.angle(E_w_prop))

    # add the zero phase with the propagated phase
    E_w_phase_add = np.unwrap(E_w_prop_phase - E_w_zero_phase)

    # inverse fourier transform
    E_t_prop = np.fft.ifft(np.fft.ifftshift(E_w_prop))

    phase = np.unwrap(np.angle(E_t_prop)) - w0 * t
    phase = phase - phase[int(len(phase)/2)]



    if plot:

        ## plot 0
        fig, ax = plt.subplots(4, 1, figsize=(8, 10))
        ax[0].plot(t, np.real(E_t), color='blue', label='real E(t)')
        ax[0].plot(t, np.imag(E_t), color='red', label='imag E(t)')
        phi_initial = np.unwrap(np.angle(E_t))
        axtwin = ax[0].twinx()
        axtwin.plot(t, phi_initial, color='green', label='$\phi_0 (t)$')
        ax[0].legend(loc=1)
        axtwin.legend(loc=2)
        ax[0].set_xlim(-30e-15, 30e-15)

        ## plot 1
        ax[1].plot(w, np.real(E_w), color='blue', label='real $E(\omega)$')
        ax[1].plot(w, np.imag(E_w), color='red', label='imag $E(\omega)$')
        ax[1].plot(w, np.abs(E_w), color='orange', label='|$E(\omega)$|')
        ax[1].set_xlim(0, 8e15)
        ax[1].legend(loc=1)
        ax_twin = ax[1].twinx()
        minindex = np.argmin(np.abs(w - 0))
        maxindex = np.argmin(np.abs(w - 8e15))
        plotspace = (k * z)[minindex:maxindex]
        ax_twin.set_ylim(np.min(plotspace), np.max(plotspace))
        init_w_phase = np.unwrap(np.angle(E_w))
        ax_twin.plot(w, k, color='purple', label='$k(\omega)$')
        ax_twin.plot(w, k * z, color='green', label='$\phi (\omega)$ applied')
        ax_twin.legend(loc=2)


        ## plot 2
        ax[2].plot(w, np.real(E_w_prop), color='blue')
        ax[2].plot(w, np.imag(E_w_prop), color='red')
        ax[2].plot(w, np.abs(E_w), color='orange', label='|$E(\omega)$|')
        ax_twin = ax[2].twinx()
        phase_w_prop = np.unwrap(np.angle(E_w_prop))
        ax_twin.plot(w, np.unwrap(phase_w_prop - init_w_phase), color='green')
        plotspace = np.unwrap(phase_w_prop - init_w_phase)[minindex:maxindex]
        ax_twin.set_ylim(np.min(plotspace), np.max(plotspace))
        ax[2].set_xlim(0, 8e15)

        ## plot 3
        ax[3].plot(t, np.real(E_t_prop), color='blue')
        ax[3].plot(t, np.imag(E_t_prop), color='red')
        ax[3].set_xlim(-30e-15, 30e-15)
        ax[3].set_ylabel('E(t)', color='blue')
        ax_twin = ax[3].twinx()

        minindex = np.argmin(np.abs(t - -3e-14))
        maxindex = np.argmin(np.abs(t - 3e-14))

        plotspace = phase[minindex:maxindex]
        ax_twin.set_ylim(np.min(plotspace), np.max(plotspace))

        ax_twin.plot(t, phase, color='green')
        ax_twin.set_ylabel('$\phi (t)$', color='green')

        # plt.show()

    return E_t_prop, t, w, dt, w0


def plot_frog(E, t, w, dt, w0, plot):
    phase = np.unwrap(np.angle(E)) - w0 * t
    phase = phase - phase[int(len(phase) / 2)]

    # calculate padded E matrix
    E_padded = np.pad(E[::-1], (0, len(E) - 1), mode='constant')

    # generate toeplitz matrix
    toeplitzmatrix = np.tril(toeplitz(E_padded, E))

    # generate t matrix
    tmatrix = np.array([E, ] * len(E_padded))

    # multiply them together
    delaymappedE = toeplitzmatrix * tmatrix

    # this is the 0 delay E_t
    # delayzero = int((len(E_padded) - 1) / 2)

    def ft_and_shift(row):
        return np.fft.fftshift(np.fft.fft(row))

    ft_delaymapped_E = np.apply_along_axis(ft_and_shift, axis=1, arr=delaymappedE)

    rangevector = np.array(range(delaymappedE.shape[0]))
    rangevector = rangevector - rangevector[-1] / 2

    # construct tau
    tau = rangevector * dt

    frogtrace = np.transpose(np.abs(ft_delaymapped_E)) ** 2

    # construct E_w
    # E_w = np.fft.fftshift(np.fft.fft(E))
    # find centroid in frequency space
    # E_w_index = np.array(range(len(E_w)))
    # centroid = (1 / len(E_w)) * np.sum(E_w_index * np.abs(E_w))
    # centroid = int(centroid)
    # w0 = w[centroid]

    taumin_index = np.argmin(np.abs(tau - -0.5e-13))
    taumax_index = np.argmin(np.abs(tau - 0.5e-13))

    w_min_index = np.argmin(np.abs(w - 0.7e16))
    w_max_index = np.argmin(np.abs(w - 0.9e16))


    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        # plot FROG trace
        ax[0].pcolormesh(tau, w, np.transpose(np.abs(ft_delaymapped_E))**2, cmap='jet')
        ax[0].set_ylim(0.7e16, 0.9e16)
        ax[0].set_xlim(-0.5e-13, 0.5e-13)
        ax[0].set_xlabel(r'$\tau$')
        ax[0].set_ylabel('$\omega$')

        ax[1].plot(t, np.abs(E), color='blue', linestyle='dashed')
        ax[1].plot(t, np.real(E), color='blue', label='real E(t)')
        ax[1].plot(t, np.imag(E), color='red', label='imag E(t)')
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('E(t)')
        ax[1].set_xlim(-50e-15, 50e-15)
        ax[1].legend(loc=2)
        axtwin = ax[1].twinx()
        minindex = np.argmin(np.abs(t - -3e-14))
        maxindex = np.argmin(np.abs(t - 3e-14))
        plotspace = phase[minindex:maxindex]
        axtwin.set_ylim(np.min(plotspace), np.max(plotspace))
        axtwin.plot(t, phase, color='green', label=r'$\phi (t)$')
        axtwin.legend(loc=4)

        plt.figure(999)

        plt.pcolormesh(frogtrace[w_min_index:w_max_index, taumin_index:taumax_index], cmap='jet')

        plt.figure(995)
        plt.pcolormesh(tau, w, frogtrace)



    # return an image of the FROG trace


    # find index where tau is closest to min max values



    return frogtrace[w_min_index:w_max_index, taumin_index:taumax_index], tau[taumin_index:taumax_index],\
           w[w_min_index:w_max_index]


def generateE_phi_vector(plot, phi_w):

    # N = 2000
    N = len(phi_w)
    dt = 0.3e-15
    df = 1 / (dt * N)

    # time axis
    t = dt * np.linspace(-N/2, N/2, N)

    # frequency axis
    w = 2 * np.pi * df * np.linspace(-N/2, N/2, N)

    # define pulse in time
    tau = 5e-15
    w0 = 4e15 # rad/s

    E_t = np.exp(1j * w0 * t) * np.exp(-t**2/tau**2)

    E_w = np.fft.fftshift(np.fft.fft(E_t))

    # get zero phase
    E_w_zero_phase = np.unwrap(np.angle(E_w))

    # apply transfer function
    z = 1  # distance
    E_w_prop = E_w * np.exp(1j * phi_w)

    # calculate spectral phase
    E_w_prop_phase = np.unwrap(np.angle(E_w_prop))

    # add the zero phase with the propagated phase
    E_w_phase_add = np.unwrap(E_w_prop_phase - E_w_zero_phase)

    # inverse fourier transform
    E_t_prop = np.fft.ifft(np.fft.ifftshift(E_w_prop))

    phase = np.unwrap(np.angle(E_t_prop)) - w0 * t
    phase = phase - phase[int(len(phase)/2)]



    if plot:

        ## plot 0
        fig, ax = plt.subplots(4, 1, figsize=(8, 10))
        ax[0].plot(t, np.real(E_t), color='blue', label='real E(t)')
        ax[0].plot(t, np.imag(E_t), color='red', label='imag E(t)')
        phi_initial = np.unwrap(np.angle(E_t))
        axtwin = ax[0].twinx()
        axtwin.plot(t, phi_initial, color='green', label='$\phi_0 (t)$')
        ax[0].legend(loc=1)
        axtwin.legend(loc=2)
        ax[0].set_xlim(-30e-15, 30e-15)

        ## plot 1
        ax[1].plot(w, np.real(E_w), color='blue', label='real $E(\omega)$')
        ax[1].plot(w, np.imag(E_w), color='red', label='imag $E(\omega)$')
        ax[1].plot(w, np.abs(E_w), color='orange', label='|$E(\omega)$|')
        ax[1].set_xlim(0, 8e15)
        ax[1].legend(loc=1)
        ax_twin = ax[1].twinx()
        minindex = np.argmin(np.abs(w - 0))
        maxindex = np.argmin(np.abs(w - 8e15))
        plotspace = (phi_w)[minindex:maxindex]
        ax_twin.set_ylim(np.min(plotspace), np.max(plotspace))
        init_w_phase = np.unwrap(np.angle(E_w))
        # ax_twin.plot(w, k, color='purple', label='$k(\omega)$')
        ax_twin.plot(w, phi_w, color='green', label='$\phi (\omega)$ applied')
        ax_twin.legend(loc=2)


        ## plot 2
        ax[2].plot(w, np.real(E_w_prop), color='blue')
        ax[2].plot(w, np.imag(E_w_prop), color='red')
        ax[2].plot(w, np.abs(E_w), color='orange', label='|$E(\omega)$|')
        ax_twin = ax[2].twinx()
        phase_w_prop = np.unwrap(np.angle(E_w_prop))
        ax_twin.plot(w, np.unwrap(phase_w_prop - init_w_phase), color='green')
        plotspace = np.unwrap(phase_w_prop - init_w_phase)[minindex:maxindex]
        ax_twin.set_ylim(np.min(plotspace), np.max(plotspace))
        ax[2].set_xlim(0, 8e15)

        ## plot 3
        ax[3].plot(t, np.real(E_t_prop), color='blue')
        ax[3].plot(t, np.imag(E_t_prop), color='red')
        ax[3].set_xlim(-30e-15, 30e-15)
        ax[3].set_ylabel('E(t)', color='blue')
        ax_twin = ax[3].twinx()

        minindex = np.argmin(np.abs(t - -3e-14))
        maxindex = np.argmin(np.abs(t - 3e-14))

        plotspace = phase[minindex:maxindex]
        ax_twin.set_ylim(np.min(plotspace), np.max(plotspace))

        ax_twin.plot(t, phase, color='green')
        ax_twin.set_ylabel('$\phi (t)$', color='green')

        # plt.show()

    return E_t_prop, t, w, dt, w0


def generate_phi_w(N, nodes, amplitude):

    phi_index = np.array(range(N))

    # phi_nodes_indexes = phi_index[::nodes]
    phi_nodes_indexes = np.linspace(phi_index[0], phi_index[-1], nodes)

    phi_nodes = amplitude * np.random.rand(1, len(phi_nodes_indexes))

    # interpolate to the larger index
    f = interpolate.interp1d(phi_nodes_indexes, phi_nodes, kind='cubic')

    phi_w = f(phi_index).reshape(-1)

    return phi_w


if __name__ == '__main__':

    # E, t, w, dt, w0 = generateE(plot=True, GVD=1e-30, TOD=15e-45)
    #
    #
    # plot_frog(E=E, t=t, w=w,  dt=dt, w0=w0, plot=True)
    #

    phi_w = generate_phi_w(N=600, nodes=100, amplitude=3)

    E, t, w, dt, w0 = generateE_phi_vector(plot=True, phi_w=phi_w)

    frogtrace, tau, w = plot_frog(E=E, t=t, w=w, dt=dt, w0=w0, plot=True)

    # plt.figure(98)

    # plt.pcolormesh(tau, w, frogtrace, cmap='jet')

    plt.show()
    #ss



