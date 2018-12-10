import numpy as np
import matplotlib.pyplot as plt



def define_E(N=512, tmax=200e-15, gdd=0.0, tod=0.0, plotting=False):

    dt = 2 * tmax / N

    t = dt * np.arange(-N/2, N/2, 1)

    df = 1 / (N * dt)

    f = df * np.arange(-N/2, N/2, 1)
    f0 = 1.5e14

    tau = 10e-15
    E_t = np.exp(-t**2 / tau**2) * np.exp(1j * 2*np.pi * f0 * t)

    E_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t)))


    # define dispersion curve
    k = 0.5 * gdd * 2*np.pi * (f - f0)**2 + (1/6) * tod * 2*np.pi * (f - f0)**3
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
        ax.plot(t, np.real(E_t), color='blue')

        # plot E_f
        ax = fig.add_subplot(gs[1,:])
        ax.plot(f, np.real(E_f), color='blue')
        ax.plot(f, np.imag(E_f) , color='red')

        # plot k(w)
        ax = fig.add_subplot(gs[2,:])
        ax.plot(f, k, color='purple')

        # plot E_F_PROP
        ax = fig.add_subplot(gs[3,:])
        ax.plot(f, np.real(E_f_prop), color='blue')
        ax.plot(f, np.imag(E_f_prop), color='red')

        # plot E_t_prop
        ax = fig.add_subplot(gs[4,:])
        ax.plot(t, np.real(E_t_prop), color='blue')
        # ax.plot(t, np.imag(E_t_prop), color='red')
        ax.plot(t, np.abs(E_t_prop), color='black')


        # make nice plot
        fig = plt.figure(figsize=(9, 6))
        gs = fig.add_gridspec(2, 2)

        # plot spectral
        min_x, max_x = 256+15, -150
        ax = fig.add_subplot(gs[:, 1])
        spec = np.abs(E_f_prop)[min_x:max_x]
        spec = spec / np.max(spec)
        ax.plot((f*1e-14)[min_x:max_x], spec, color='black', label='$|E(f)|$')
        ax.set_xlim((f*1e-14)[min_x], (f*1e-14)[max_x])
        ax.set_ylim(0, 1)
        axtwin = ax.twinx()
        phase = np.unwrap(np.angle((E_f_prop[min_x:max_x])))
        phase = phase - np.min(phase)
        axtwin.plot((f*1e-14)[min_x:max_x],
                    phase, color='green',
                    label='Real $E(t)$')
        ax.legend(loc=1)

        # plot in time
        min_x, max_x = 150, -150
        ax = fig.add_subplot(gs[:, 0])
        ax.plot((t * 1e15)[min_x:max_x], np.real(E_t_prop)[min_x:max_x], color='blue', label='Real $E(t)$')
        ax.plot((t * 1e15)[min_x:max_x], np.abs(E_t_prop)[min_x:max_x], color='black', label='$|E(t)|$', linestyle='dashed')
        gdd_num = round(gdd * 1e30, 10)
        tod_num = round(tod * 1e45, 10)
        ax.text(0.05, 0.89, 'GDD: {} [$fs^2$]\nTOD: {} [$fs^3$]'.format(gdd_num, tod_num), backgroundcolor='white',
                transform=ax.transAxes)
        ax.set_xlim((t * 1e15)[min_x], (t * 1e15)[max_x])
        ax.set_ylim(-1, 1)
        axtwin = ax.twinx()
        phase = np.unwrap(np.angle((E_t_prop[min_x:max_x])))-2*np.pi*f0*t[min_x:max_x]
        phase = phase - np.min(phase)
        axtwin.plot((t * 1e15)[min_x:max_x],
                    phase, color='green',
                    label='Real $E(t)$')
        ax.legend(loc=1)


    return E_f_prop, E_t_prop, t, f




def construct_frog_trace(E_t, E_f, t, f, plotting=False):

    delaymatrix = np.exp(1j * 2 * np.pi * t.reshape(-1, 1) * f.reshape(1, -1))

    delayed_E_f = E_f.reshape(1, -1) * delaymatrix

    delayed_E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(delayed_E_f, axes=1), axis=1), axes=1)

    product = delayed_E_t * E_t.reshape(1, -1)

    product_ft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(product, axes=1), axis=1), axes=1)

    trace = np.transpose(np.abs(product_ft)**2)


    if plotting:

        fig = plt.figure()
        gs = fig.add_gridspec(5, 2)

        ax = fig.add_subplot(gs[:, :])
        ax.pcolormesh(t, f, trace, cmap='jet')
        ax.set_ylabel('frequency')
        ax.set_xlabel('delay')



    return  trace


def plot_E_t_and_trace(t, f, E_t, trace):

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[:,0])
    min_x, max_x = 150, -150
    ax.plot((t*1e15)[min_x:max_x], np.real(E_t)[min_x:max_x], color='blue', label='Real $E(t)$')
    # ax.plot((t*1e15)[min_x:max_x], np.imag(E_t)[min_x:max_x], color='red', label='Imag $E(t)$')
    ax.plot((t*1e15)[min_x:max_x], np.abs(E_t)[min_x:max_x], color='black', label='$|E(t)|$', linestyle='dashed')
    # ax.plot(t*1e15, np.imag(E_t), color='red')
    ax.set_xlabel('time [fs]')
    ax.set_ylabel('Electric Field\n[arbitrary units]')
    gdd_num = round(gdd*1e30, 10)
    tod_num = round(tod*1e45, 10)
    ax.text(0.05, 0.89, 'GDD: {} [$fs^2$]\nTOD: {} [$fs^3$]'.format(gdd_num, tod_num), backgroundcolor='white', transform=ax.transAxes)
    ax.set_ylim(-1,1)
    ax.set_xlim((t*1e15)[min_x],(t*1e15)[max_x])
    ax.set_title('Electric Field')
    ax.legend(loc=1)

    ax = fig.add_subplot(gs[:, 1])
    min_x, max_x = 150, -150
    min_y, max_y = 300, -60
    ax.pcolormesh((t*1e15)[min_x:max_x], (f*1e-14)[min_y:max_y], trace[min_y:max_y:, min_x:max_x], cmap='jet')
    ax.set_xlabel('delay [fs]')
    ax.set_ylabel('frequency [$10^{14}$Hz]')
    ax.set_title('FROG trace')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.3, wspace=0.1, left=0.10, right=0.90)






if __name__ == "__main__":

    # gdd = 700e-30 # s**2
    gdd = 0 # s**2
    tod = 25000e-45 # s**3
    # tod = 0



    E_f, E_t, t, f = define_E(gdd=gdd, tod=tod, plotting=True)
    # E_f, E_t, t, f = define_E(plotting=False)


    trace = construct_frog_trace(E_t, E_f, t, f, plotting=False)


    plot_E_t_and_trace(t, f, E_t, trace)

    plt.show()













