import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

tmax = 10
N = 128
tau = 1
dt = 2 * tmax / N
t = dt * np.arange(-N/2, N/2, 1)
E_t = np.exp(-t**2 / tau**2)
df = 1 / (dt * N)
f = df * np.arange(-N/2, N/2, 1)

# applt random phase to E_t
num_nodes = 30
amplitude = 8
phase_nodes = amplitude * (np.random.rand(num_nodes)-0.5)
phase_nodes_f = np.linspace(f[0], f[-1], num_nodes)
phase_interp = interpolate.interp1d(phase_nodes_f, phase_nodes, kind='cubic')
phase = phase_interp(f)


E_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t)))
E_f_phase = E_f * np.exp(1j * phase)
E_t_phase = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_f_phase)))


# cosntruct delay grid
delay_grid = t.reshape(-1, 1) * f.reshape(1, -1)

#fft pulse
E_f_phase_delay = E_f_phase.reshape(1, -1) * np.exp(1j * 2 * np.pi * delay_grid)
E_t_phase_delay = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_f_phase_delay, axes=1), axis=1), axes=1)


# construct constant E_t
E_t_nodelay = E_t_phase.reshape(1, -1) * np.ones_like(E_t_phase_delay)


# integrate vertically
integral_delay = dt * np.sum(E_t_phase_delay * E_t_nodelay, axis=0)


frog = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t_phase_delay * E_t_nodelay, axes=1), axis=1), axes=1)


fig = plt.figure()
gs = fig.add_gridspec(6, 3)

ax = fig.add_subplot(gs[0,:2])
ax.plot(f, np.real(E_f_phase), color='blue')
ax.plot(f, np.imag(E_f_phase), color='red')
axtwin = ax.twinx()
axtwin.plot(f, phase, color='green')



ax = fig.add_subplot(gs[1,:2])
ax.plot(t, np.real(E_t_phase), color='blue')
ax.plot(t, np.imag(E_t_phase), color='red')
ax.plot(t, np.abs(E_t_phase), color='black')

ax = fig.add_subplot(gs[2,:2])
ax.pcolormesh(t, t, np.real(E_t_phase_delay), cmap='jet')
ax.set_ylabel(r'$\tau$')
ax.set_xlabel(r'$t$')

ax = fig.add_subplot(gs[3,:2])
ax.pcolormesh(t, t, np.real(E_t_nodelay), cmap='jet')


ax = fig.add_subplot(gs[1,2])
ax.pcolormesh(t, t, np.imag(E_t_phase_delay  * E_t_nodelay), cmap='jet')

ax = fig.add_subplot(gs[2,2])
ax.pcolormesh(t, t, np.real(E_t_phase_delay  * E_t_nodelay), cmap='jet')

ax = fig.add_subplot(gs[3,2])
ax.pcolormesh(t, t, np.abs(E_t_phase_delay  * E_t_nodelay), cmap='jet')



ax = fig.add_subplot(gs[4,2])
ax.pcolormesh(t, t, np.abs(frog)**2, cmap='jet')



ax = fig.add_subplot(gs[4,:2])
ax.plot(t, np.real(integral_delay), color='blue')
ax.plot(t, np.imag(integral_delay), color='red')
ax.plot(t, np.abs(integral_delay), color='black')
plt.show()


exit(0)





