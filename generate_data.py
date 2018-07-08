from frognet1 import generate_phi_w, generateE_phi_vector, plot_frog
import tables
import numpy as np
import matplotlib.pyplot as plt


def retrieve_data(plot_frog_bool, print_size):

    phi_w = generate_phi_w(N=2**7, nodes=30, amplitude=3)

    E, t, w, dt, w0 = generateE_phi_vector(plot=False, phi_w=phi_w)

    frogtrace, tau, w = plot_frog(E=E, t=t, w=w, dt=dt, w0=w0, plot=plot_frog_bool)

    if print_size:
        print('original size: ', frogtrace.shape)
        print('E size', E.shape)

    return E, t, frogtrace.reshape(-1)


if __name__ == '__main__':
    E, t, frogtrace_flat = retrieve_data(plot_frog_bool=False, print_size=True)

    # data for input
    E_real = np.real(E)
    E_imag = np.imag(E)

    # create file
    hdf5_file = tables.open_file('frogtrainingdata.hdf5', mode='w')
    frog_image = hdf5_file.create_earray(hdf5_file.root,
                                            'frog', tables.Float16Atom(), shape=(0, len(frogtrace_flat)))
    E_real = hdf5_file.create_earray(hdf5_file.root,
                                      'E_real', tables.Float16Atom(), shape=(0, len(E_real)))
    E_imag = hdf5_file.create_earray(hdf5_file.root,
                                      'E_imag', tables.Float16Atom(), shape=(0, len(E_imag)))
    hdf5_file.close()


    # populate file
    print('generating samples')
    n_samples = 300
    hdf5_file = tables.open_file('frogtrainingdata.hdf5', mode='a')
    for i in range(n_samples):

        E, t, frogtrace_flat = retrieve_data(plot_frog_bool=False, print_size=False)

        E_real = np.real(E)
        E_imag = np.imag(E)
        hdf5_file.root.E_real.append(E_real.reshape(1, -1))
        hdf5_file.root.frog.append(frogtrace_flat.reshape(1, -1))
        hdf5_file.root.E_imag.append(E_imag.reshape(1, -1))

        if i % 5 == 0:
            print('generating sample: ', i, ' of ', n_samples)
    hdf5_file.close()


    # open and read
    hdf5_file = tables.open_file('frogtrainingdata.hdf5', mode='r')
    index = 0

    E = hdf5_file.root.E_real[index, :] + 1j * hdf5_file.root.E_imag[index, :]
    fig, ax = plt.subplots(2, 1)
    ax[0].pcolormesh(hdf5_file.root.frog[index, :].reshape(58, 106), cmap='jet')
    ax[1].plot(t, np.abs(E), color='black', linestyle='dashed', alpha=0.5)
    ax[1].plot(t, np.real(E), color='blue')
    ax[1].plot(t, np.imag(E), color='red')

    hdf5_file.close()

    plt.show()

