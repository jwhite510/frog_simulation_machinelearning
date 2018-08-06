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

    # return square trace
    middle = int((len(tau) + 1) / 2) - 1
    upper = int(middle + (len(tau)+1) / 4)
    lower = int(middle - (len(tau)+1) / 4)

    return frogtrace[:, lower:upper], tau[lower:upper], w


def generateE_phi_vector(plot, phi_w):

    N = len(phi_w)

    tmax = 40e-15
    dt = 2 * tmax / N

    df = 1 / (dt * N)

    t = dt * np.arange(-N/2, N/2, 1)

    w = 2 * np.pi * df * np.arange(-N/2, N/2, 1)

    # for testing with GVD and TOD
    # GVD = 10e-30
    # TOD = 5e-45
    # phi_w = GVD * w**2 - TOD * w**3

    # define pulse in time
    tau = 5e-15
    w0 = 0

    E_t = np.exp(-t**2 / tau**2)

    E_w = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t)))

    E_w_prop = E_w * np.exp(1j * phi_w)

    E_t_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_w_prop)))

    # start ambiguity removal!
    #   first shift the intensity peak to time 0
    I_peak_index = np.argmax(np.abs(E_t_prop)**2)
    t_0_index = np.argmin(np.abs(t - 0.0))
    t_0 = t[t_0_index]
    steps_diff = t_0_index - I_peak_index
    E_rolled = np.roll(E_t_prop, steps_diff)
    #replace rolled points with 0
    if steps_diff > 0:
        E_rolled[:steps_diff] = 0
    if steps_diff < 0:
        E_rolled[steps_diff:] = 0

    # apply phase shift at time0
    phi_t_0 = np.unwrap(np.angle(E_rolled))[t_0_index]
    phi_t_0_constant_vec = phi_t_0 * np.ones_like(E_rolled)
    E_rolled_phi_corrected = E_rolled * np.exp(-1j * phi_t_0_constant_vec)

    # check if integral of real part on left is greater than right

    integral_left_side = dt * np.sum(np.real(E_rolled_phi_corrected[:t_0_index]))
    integral_right_side = dt * np.sum(np.real(E_rolled_phi_corrected[t_0_index+1:]))

    # make flip always by setting to one
    # integral_right_side = 1
    #######

    flip = False
    if integral_right_side > integral_left_side:
        flip=True
        flipped_final_E = np.flip(E_rolled_phi_corrected, 0)
        # roll by one, because of flip with even number of timesteps
        flipped_final_E = np.roll(flipped_final_E, 1)
        flipped_final_E[:1] = 0

    else:
        flipped_final_E = E_rolled_phi_corrected[:]

    if plot:

        # round for plotting later
        integral_right_side = round(integral_right_side, 18)
        integral_left_side = round(integral_left_side, 18)

        # fourier transform final field
        flipped_final_E_w = np.fft.fftshift(np.fft.fft(np.fft.fftshift(flipped_final_E)))

        # ft non flipped field
        E_rolled_phi_corrected_w = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_rolled_phi_corrected)))

        fig, ax = plt.subplots(6, 2, figsize=(12, 10))

        ax[0][0].plot(t, np.real(E_t), color='blue')
        ax[0][0].plot(t, np.imag(E_t), color='red')
        ax[0][0].plot(t, np.abs(E_t), color='black', linestyle='dashed', alpha=0.5)
        ax[0][0].set_ylabel('$E(t)$')
        axtwin = ax[0][0].twinx()
        axtwin.text(0.4, 1.15, 'Generation of sample $E(t)$', transform=axtwin.transAxes,
                    backgroundcolor='yellow')
        axtwin.text(0.55, 0.90, 'Initially transform limited pulse $E(t)$', transform=axtwin.transAxes,
                    backgroundcolor='white')
        axtwin.plot(t, np.unwrap(np.angle(E_t)), color='green')
        axtwin.set_ylabel('$\phi (t)$', color='green')

        ax[1][0].plot(w, np.real(E_w), color='blue')
        ax[1][0].plot(w, np.imag(E_w), color='red')
        ax[1][0].plot(w, np.abs(E_w), color='black', linestyle='dashed', alpha=0.5)
        ax[1][0].set_ylabel('$E(\omega)$')
        axtwin = ax[1][0].twinx()
        axtwin.text(0.55, 0.90, 'Fourier transformed \n random phase generated', transform=axtwin.transAxes,
                    backgroundcolor='white')
        axtwin.plot(w, np.unwrap(np.angle(E_w)), color='green')
        axtwin.plot(w, phi_w, color='green', linestyle='dashed', label='random phase $\phi(\omega)$')
        axtwin.set_ylabel('$\phi (\omega)$', color='green')
        axtwin.legend(loc=2)
        ax[2][0].plot(w, np.real(E_w_prop), color='blue')
        ax[2][0].plot(w, np.imag(E_w_prop), color='red')
        ax[2][0].plot(w, np.abs(E_w_prop), color='black', linestyle='dashed', alpha=0.5)
        ax[2][0].set_ylabel('$E(\omega)$')
        axtwin = ax[2][0].twinx()
        axtwin.text(0.55, 0.90, 'Random phase applied', transform=axtwin.transAxes,
                    backgroundcolor='white')
        axtwin.plot(w, np.unwrap(np.angle(E_w_prop)), color='green')
        axtwin.set_ylabel('$\phi (\omega)$', color='green')

        ax[3][0].plot(t, np.real(E_t_prop), color='blue')
        ax[3][0].plot(t, np.imag(E_t_prop), color='red')
        ax[3][0].plot(t, np.abs(E_t_prop), color='black', linestyle='dashed', alpha=0.5)
        ax[3][0].set_ylabel('$E(t)$')
        axtwin = ax[3][0].twinx()
        axtwin.text(0.55, 0.90, 'Fourier transform back to time', transform=axtwin.transAxes,
                    backgroundcolor='white')
        axtwin.plot(t, np.unwrap(np.angle(E_t_prop)), color='green')
        axtwin.set_ylabel('$\phi (t)$', color='green')

        ax[0][1].text(0.4, 1.15, 'Removing ambuguities', transform=ax[0][1].transAxes,
                      backgroundcolor='yellow')
        ax[0][1].plot(t, np.real(E_t_prop), color='blue')
        ax[0][1].plot(t, np.imag(E_t_prop), color='red')
        ax[0][1].plot(t, np.abs(E_t_prop)**2, color='black', label='intensity')
        ax[0][1].legend(loc=2)
        # max intensity marker
        ax[0][1].plot(t[I_peak_index], 1, marker='o', color='purple')
        ax[0][1].text(t[I_peak_index+2], 0.8, 'maximum intensity', color='purple')
        ax[0][1].text(t[0], -0.8, '$I(t)$ peak off by {} timesteps'.format(steps_diff))
        # center marker
        ax[0][1].plot([t_0, t_0], [-1, 1], color='black', alpha=0.5)
        print('steps diff: ', steps_diff)

        ax[1][1].plot(t, np.real(E_rolled), color='blue')
        ax[1][1].plot(t, np.imag(E_rolled), color='red')
        ax[1][1].plot(t, np.abs(E_rolled)**2, color='black', label='intensity')
        ax[1][1].plot([t_0, t_0], [-1, 1], color='black', alpha=0.5)
        ax[1][1].legend(loc=2)
        # plot phase
        axtwin = ax[1][1].twinx()
        axtwin.text(0.6, 0.9, 'intensity peak moved to center', backgroundcolor='white',
                    transform=axtwin.transAxes)
        axtwin.plot(t, np.unwrap(np.angle(E_rolled)), color='green')
        axtwin.set_ylabel('$\phi (t)$', color='green')
        ax[2][1].plot(t, np.real(E_rolled_phi_corrected), color='blue')
        ax[2][1].plot(t, np.imag(E_rolled_phi_corrected), color='red')
        ax[2][1].plot(t, np.abs(E_rolled_phi_corrected)**2, color='black', label='intensity')
        ax[2][1].plot([t_0, t_0], [-1, 1], color='black', alpha=0.5)
        axtwin = ax[2][1].twinx()
        axtwin.plot(t, np.unwrap(np.angle(E_rolled_phi_corrected)), color='green')
        print('t0 at t0 index: ', t[t_0_index])
        axtwin.text(0.6, 0.9, 'phase angle at t=0 set to 0', backgroundcolor='white',
                    transform=axtwin.transAxes)
        axtwin.set_ylabel('$\phi (t)$', color='green')
        ax[2][1].legend(loc=2)
        axtwin.text(-0.05, 0,
                      'left real side integral: {}'.format(integral_left_side),
                      backgroundcolor='white', transform=ax[2][1].transAxes)
        axtwin.text(-0.05, 0.22,
                      'right real side integral: {}'.format(integral_right_side),
                      backgroundcolor='white', transform=ax[2][1].transAxes)
        axtwin.text(0.6, 0.12, '------->', transform=ax[2][1].transAxes, backgroundcolor='white')

        axtwin.text(0.8, 0.1, 'Flip:', color='black', transform=ax[2][1].transAxes)
        if flip:
            axtwin.text(0.9, 0.1, 'True', color='black', transform=ax[2][1].transAxes,
                          backgroundcolor='green')
        else:
            axtwin.text(0.9, 0.1, 'False', color='black', transform=ax[2][1].transAxes,
                          backgroundcolor='red')

        ax[3][1].plot(t, np.real(flipped_final_E), color='blue')
        ax[3][1].plot(t, np.imag(flipped_final_E), color='red')
        ax[3][1].plot(t, np.abs(flipped_final_E)**2, color='black', label='intensity')
        ax[3][1].plot([t_0, t_0], [-1, 1], color='black', alpha=0.5)
        ax[3][1].legend(loc=2)
        axtwin = ax[3][1].twinx()
        axtwin.plot(t, np.unwrap(np.angle(flipped_final_E)), color='green')
        axtwin.set_ylabel('$\phi (t)$', color='green')
        axtwin.text(0.6, 0.9, 'flipped, final field for testing', transform=ax[3][1].transAxes,
                    backgroundcolor='white')

        # plot the spectrum and spectral phase of E(t) and time reversed E(t)
        crop_phase = 15

        ax[4][1].plot(w, np.real(flipped_final_E_w), color='blue')
        ax[4][1].plot(w, np.imag(flipped_final_E_w), color='red')
        ax[4][1].plot(w, np.abs(flipped_final_E_w), color='black', linestyle='dashed',
                      label='abs')
        ax[4][1].legend(loc=2)
        axtwin = ax[4][1].twinx()
        axtwin.plot(w[crop_phase:-crop_phase], np.unwrap(np.angle(flipped_final_E_w))[crop_phase:-crop_phase], color='green')
        axtwin.text(0.55, 0.9, 'fourier transform of flipped field (if flipped)',
                    transform=axtwin.transAxes, backgroundcolor='white')

        ax[5][1].plot(w, np.real(E_rolled_phi_corrected_w), color='blue')
        ax[5][1].plot(w, np.imag(E_rolled_phi_corrected_w), color='red')
        ax[5][1].plot(w, np.abs(E_rolled_phi_corrected_w), color='black', linestyle='dashed',
                      label='abs')
        ax[5][1].legend(loc=2)
        axtwin = ax[5][1].twinx()
        axtwin.plot(w[crop_phase:-crop_phase], np.unwrap(np.angle(E_rolled_phi_corrected_w))[crop_phase:-crop_phase], color='green')
        axtwin.text(0.55, 0.9, 'fourier transform of non - flipped field',
                    transform=axtwin.transAxes, backgroundcolor='white')

        # add numbers to plots
        for i, number in zip(range(4), [1, 2, 3, 4]):
            ax[i][0].text(0, 1, str(number), transform=ax[i][0].transAxes,
                          backgroundcolor='red')

        for i, number in zip(range(4), [5, 6, 7, 8]):
            ax[i][1].text(0, 1, str(number), transform=ax[i][1].transAxes,
                          backgroundcolor='red')



    return flipped_final_E, t, w, dt, w0


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

    phi_w = generate_phi_w(N=2**6, nodes=15, amplitude=3)

    # output 64 time step E
    E, t, w, dt, w0 = generateE_phi_vector(plot=False, phi_w=phi_w)
    # print(np.shape(E))
    # output 64x64 frog trace
    frogtrace, tau, w = plot_frog(E=E, t=t, w=w, dt=dt, w0=w0, plot=False)
    # print(np.shape(frogtrace))
    # print(np.shape(tau))

    plt.figure(99)
    plt.pcolormesh(frogtrace, cmap='jet')

    plt.figure(101)
    plt.pcolormesh(tau, w, frogtrace, cmap='jet')

    fig, ax = plt.subplots()
    ax.plot(t, np.real(E), color='blue')
    ax.plot(t, np.imag(E), color='red')

    plt.show()



