import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches




N = 512
tmax = 100e-15
dt = 2 * tmax / N

t = dt * np.arange(-N/2, N/2)

df = 1 / (dt * N)

f = df * np.arange(-N/2, N/2)

tau = 10e-15
f0 = 10e13

E_t = np.exp(-t**2 / tau**2) * np.exp(1j * 2*np.pi * f0 * t)
E_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t)))

GDD = 1500e-30
TOD = 15000e-45
k = GDD * (f - f0)**2 + TOD * (f - f0)**3
z = 1

E_f_prop = E_f * np.exp(1j * k * z)
E_t_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_f_prop)))





fig, ax = plt.subplots(1, 3, figsize=(10,4))
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, hspace=0.1, wspace=0.2)


ax[0].plot(np.real(E_t), color='black', label='$E(t)$')
ax[0].legend(loc=0)
ax[0].set_xlabel(r'time $\rightarrow$')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].text(0.1, 0.8, "z: 0m", transform=ax[0].transAxes)


f_crop_l = 250
f_crop_r = 210
lns1 = ax[1].plot(f[f_crop_l:-f_crop_r], np.abs(E_f)[f_crop_l:-f_crop_r], label='$|U(\omega)|$', color='black')
axtwin = ax[1].twinx()
lns2 = axtwin.plot(f[f_crop_l:-f_crop_r], k[f_crop_l:-f_crop_r], color='purple', label='$k(\omega)$')
lns = lns1 + lns2
labels = [l.get_label() for l in lns]
axtwin.legend(lns, labels, loc=1)
axtwin.set_yticks([])
ax[1].set_xlabel(r'$\omega \rightarrow$')
ax[1].set_xticks([])
ax[1].set_yticks([])




arrow = matplotlib.patches.FancyArrowPatch(
    (0.30, 0.2), (0.70, 0.2), transform=fig.transFigure,
    arrowstyle='simple', mutation_scale=40, alpha=1,
    fc='r',
    connectionstyle="arc3,rad=0.2"

)




fig.patches.append(arrow)
fig.text(0.5, 0.08, '$U(\omega)e^{i k (\omega) z}$', transform=fig.transFigure, backgroundcolor='red', ha='center', size=12)
fig.text(0.4, 0.11, '$FFT$', transform=fig.transFigure, backgroundcolor='red', ha='center', size=12)
fig.text(0.6, 0.11, '$FFT^{-1}$', transform=fig.transFigure, backgroundcolor='red', ha='center', size=12)


ax[2].plot(np.real(E_t_prop), color='black', label='$E(t)$')
ax[2].set_xlabel(r'time $\rightarrow$')
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].text(0.1, 0.8, "z: 1m", transform=ax[2].transAxes)
ax[2].legend(loc=0)

plt.savefig("./beampropagation.png")

plt.show()









