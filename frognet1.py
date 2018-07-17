import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy import interpolate


def plot_frog(E, t, w, dt, w0, plot):

    # calculate padded E matrix
    E_padded = np.pad(E[::-1], (0, len(E) - 1), mode='constant')

    # generate toeplitz matrix
    toeplitzmatrix = np.tril(toeplitz(E_padded, E))

    # generate t matrix
    tmatrix = np.array([E, ] * len(E_padded))

    # multiply them together
    delaymappedE = toeplitzmatrix * tmatrix


    def ft_and_shift(row):
        return np.fft.fftshift(np.fft.fft(np.fft.fftshift(row)))

    ft_delaymapped_E = np.apply_along_axis(ft_and_shift, axis=1, arr=delaymappedE)

    # plt.figure(6)
    # plt.plot(w, ft_delaymapped_E[127])

    rangevector = np.array(range(delaymappedE.shape[0]))
    rangevector = rangevector - rangevector[-1] / 2

    # construct tau
    tau = rangevector * dt

    frogtrace = np.transpose(np.abs(ft_delaymapped_E)) ** 2

    taumax = 5e-14
    w_max = 1.5e15

    taumin_index = np.argmin(np.abs(tau - -taumax))
    taumax_index = np.argmin(np.abs(tau - taumax))

    w_min_index = np.argmin(np.abs(w - -w_max))
    w_max_index = np.argmin(np.abs(w - w_max))


    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        # plot FROG trace
        ax[0].pcolormesh(tau, w, frogtrace, cmap='jet')
        ax[0].set_ylim(-w_max, w_max)
        ax[0].set_xlim(-taumax, taumax)
        ax[0].set_xlabel(r'$\tau$')
        ax[0].set_ylabel('$\omega$')

        ax[1].plot(t, np.real(E), color='blue', label='real E(t)')
        ax[1].plot(t, np.imag(E), color='red', label='imag E(t)')
        ax[1].plot(t, np.abs(E), color='black', linestyle='dashed', alpha=0.5, label='|E(t)|')
        ax[1].legend(loc=2)
        axtwin = ax[1].twinx()
        axtwin.plot(t, np.unwrap(np.angle(E)), color='green', label='$\phi (t)$')
        axtwin.legend(loc=1)

        # plt.figure(999)
        # plt.pcolormesh(frogtrace[w_min_index:w_max_index, taumin_index:taumax_index], cmap='jet')

    return frogtrace[w_min_index:w_max_index, taumin_index:taumax_index], tau[taumin_index:taumax_index],\
           w[w_min_index:w_max_index]


def generateE_phi_vector(plot, phi_w):

    N = len(phi_w)

    tmax = 60e-15
    dt = 2 * tmax / N

    df = 1 / (dt * N)

    t = dt * np.arange(-N/2, N/2, 1)

    w = 2 * np.pi * df * np.arange(-N/2, N/2, 1)

    # for testing with GVD and TOD
    # GVD = 0e-30
    # TOD = 10e-45
    # phi_w = GVD * w**2 + TOD * w**3

    # define pulse in time
    tau = 5e-15
    w0 = 0

    E_t = np.exp(-t**2 / tau**2)

    E_w = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t)))

    E_w_prop = E_w * np.exp(1j * phi_w)

    E_t_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_w_prop)))

    if plot:

        fig, ax = plt.subplots(4, 1, figsize=(8, 10))

        ax[0].plot(t, np.real(E_t), color='blue')
        ax[0].plot(t, np.imag(E_t), color='red')
        ax[0].plot(t, np.abs(E_t), color='black', linestyle='dashed', alpha=0.5)
        ax[0].set_ylabel('$E(t)$')
        axtwin = ax[0].twinx()
        axtwin.plot(t, np.unwrap(np.angle(E_t)), color='green')
        axtwin.set_ylabel('$\phi (t)$', color='green')

        ax[1].plot(w, np.real(E_w), color='blue')
        ax[1].plot(w, np.imag(E_w), color='red')
        ax[1].plot(w, np.abs(E_w), color='black', linestyle='dashed', alpha=0.5)
        ax[1].set_ylabel('$E(\omega)$')
        axtwin = ax[1].twinx()
        axtwin.plot(w, np.unwrap(np.angle(E_w)), color='green')
        axtwin.plot(w, phi_w, color='green', linestyle='dashed')
        axtwin.set_ylabel('$\phi (\omega)$', color='green')

        ax[2].plot(w, np.real(E_w_prop), color='blue')
        ax[2].plot(w, np.imag(E_w_prop), color='red')
        ax[2].plot(w, np.abs(E_w_prop), color='black', linestyle='dashed', alpha=0.5)
        ax[2].set_ylabel('$E(\omega)$')
        axtwin = ax[2].twinx()
        axtwin.plot(w, np.unwrap(np.angle(E_w_prop)), color='green')
        axtwin.set_ylabel('$\phi (\omega)$', color='green')

        ax[3].plot(t, np.real(E_t_prop), color='blue')
        ax[3].plot(t, np.imag(E_t_prop), color='red')
        ax[3].plot(t, np.abs(E_t_prop), color='black', linestyle='dashed', alpha=0.5)
        ax[3].set_ylabel('$E(t)$')
        axtwin = ax[3].twinx()
        axtwin.plot(t, np.unwrap(np.angle(E_t_prop)), color='green')
        axtwin.set_ylabel('$\phi (t)$', color='green')


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

    phi_w = generate_phi_w(N=2**7, nodes=30, amplitude=3)

    E, t, w, dt, w0 = generateE_phi_vector(plot=True, phi_w=phi_w)

    frogtrace, tau, w = plot_frog(E=E, t=t, w=w, dt=dt, w0=w0, plot=True)

    plt.figure(98)
    plt.pcolormesh(tau, w, frogtrace, cmap='jet')

    plt.figure(99)
    plt.plot(t, np.abs(E))

    plt.show()



