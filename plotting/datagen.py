import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


N=2**6
nodes=15
amplitude=5


phi_index = np.array(range(N))

# phi_nodes_indexes = phi_index[::nodes]
phi_nodes_indexes = np.linspace(phi_index[0], phi_index[-1], nodes)
phi_nodes = amplitude * np.random.rand(1, len(phi_nodes_indexes))

# interpolate to the larger index
f = interpolate.interp1d(phi_nodes_indexes, phi_nodes, kind='cubic')
phi_w = f(phi_index).reshape(-1)


# generate pulse in time
tmax = 40e-15
dt = 2 * tmax / N
df = 1 / (dt * N)
t = dt * np.arange(-N/2, N/2, 1)
w = 2 * np.pi * df * np.arange(-N/2, N/2, 1)

tau = 5e-15
E_t = np.exp(-t**2 / tau**2)
E_w = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t)))

E_w_prop = E_w * np.exp(1j * phi_w)


# ft back to temporal domain
E_t_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_w_prop)))

fig, ax = plt.subplots(4, 1, figsize=(5, 6))
ax[1].plot(phi_index, phi_w, color='green', label=r'$\phi(\omega)$')
ax[1].plot(phi_nodes_indexes, phi_nodes.reshape(-1), '*', color='orange', label='Nodes')
ax[1].legend(loc=2)

ax[0].plot(w, np.real(E_w), color='blue', label='real $U(\omega)$')
ax[0].plot(w, np.imag(E_w), color='red', label='imag $U(\omega)$')
ax[0].legend(loc=2)


ax[2].plot(w, phi_w, color='green', linestyle='dashed')
axtwin = ax[2].twinx()
axtwin.plot(w, np.real(E_w_prop), color='blue', label='real $E(\omega)$')
axtwin.plot(w, np.imag(E_w_prop), color='red', label='imag $E(\omega)$')
axtwin.set_yticks([])
axtwin.legend(loc=2)


ax[3].plot(t, np.real(E_t_prop), color='blue', label='real $E(t)$')
ax[3].plot(t, np.imag(E_t_prop), color='red', label='imag $E(t)$')
ax[3].legend(loc=2)





for i, letter in zip(range(4), ['a)', 'b)', 'c)', 'd)']):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].text(-0.05, 0.9, letter, transform=ax[i].transAxes)


plt.savefig('/home/zom/Documents/MAKETEX_PROJECTS/research_paper/phasegen.png')


plt.show()

