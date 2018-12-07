import numpy as np
import matplotlib.pyplot as plt



def define_E(N=64, tmax=50e-15, gdd=1000e-30, tod=20000e-45, plotting=False):

    dt = 2 * tmax / N

    t = dt * np.arange(-N/2, N/2, 1)

    df = 1 / (N * dt)

    f = df * np.arange(-N/2, N/2, 1)
    # f0 = 2e14
    f0 = 0

    tau = 10e-15
    E_t = np.exp(-t**2 / tau**2) * np.exp(1j * 2 *np.pi * f0 * t)

    E_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t)))


    # define dispersion curve
    k = gdd * (f - f0)**2 + tod * (f - f0)**3
    z = 1 # meter


    # apply phase
    E_f_prop = E_f * np.exp(1j * k * z)

    # reverse ft
    E_t_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_f_prop)))


    if plotting:

        fig = plt.figure()
        gs = fig.add_gridspec(5,2)

        # plot E_t
        ax = fig.add_subplot(gs[0,:])
        ax.plot(t, np.real(E_t))

        # plot E_f
        ax = fig.add_subplot(gs[1,:])
        ax.plot(t, np.real(E_f), color='blue')
        ax.plot(t, np.imag(E_f) , color='red')

        # plot k(w)
        ax = fig.add_subplot(gs[2,:])
        ax.plot(f, k)

        # plot E_F_PROP
        ax = fig.add_subplot(gs[3,:])
        ax.plot(f, np.real(E_f_prop), color='blue')
        ax.plot(f, np.imag(E_f_prop), color='red')

        # plot E_t_prop
        ax = fig.add_subplot(gs[4,:])
        ax.plot(t, np.real(E_t_prop), color='blue')
        # ax.plot(t, np.imag(E_t_prop), color='red')
        ax.plot(t, np.abs(E_t_prop), color='black')


    return E_f_prop, E_t_prop, t, f




def construct_frog_trace(E_t, E_f, t, f, plotting=True):

    delaymatrix = np.exp(1j * 2 * np.pi * t.reshape(-1, 1) * f.reshape(1, -1))

    delayed_E_f = E_f.reshape(1, -1) * delaymatrix

    delayed_E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(delayed_E_f, axes=1), axis=1), axes=1)

    product = delayed_E_t * E_t.reshape(1, -1)

    product_ft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(product)))

    trace = np.transpose(np.abs(product_ft)**2)


    if plotting:

        fig = plt.figure()
        gs = fig.add_gridspec(5, 2)

        ax = fig.add_subplot(gs[:, :])
        ax.pcolormesh(t, f, trace, cmap='jet')
        ax.set_ylabel('frequency')
        ax.set_xlabel('delay')



    return  trace






if __name__ == "__main__":



    # E_f, E_t, t, f = define_E(gdd=0, tod=0, plotting=True)
    E_f, E_t, t, f = define_E(plotting=True)


    trace = construct_frog_trace(E_t, E_f, t, f)

    plt.show()













