import numpy as np
import matplotlib.pyplot as plt
from frognet1 import generate_phi_w, generateE_phi_vector, plot_frog


# generate original pulse
phi_w = generate_phi_w(N=2**6, nodes=30, amplitude=3)
Eoriginal, t, w, dt, w0 = generateE_phi_vector(plot=False, phi_w=phi_w)
frogtraceOriginal, tau, w = plot_frog(E=Eoriginal, t=t, w=w, dt=dt, w0=w0, plot=False)

#translation
Etranslated = np.roll(Eoriginal, 10)
Etranslated[:10] = 0
frogtraceTranslated, tau, w = plot_frog(E=Etranslated, t=t, w=w, dt=dt, w0=w0, plot=False)

# constant phase shift
Econstphase = Eoriginal * np.exp(1j * 0.25 * np.pi)
frogtraceconstphase, tau, w = plot_frog(E=Econstphase, t=t, w=w, dt=dt, w0=w0, plot=False)

# conjugate flip
Eflip = np.flip(np.real(Eoriginal) - 1j * np.imag(Eoriginal), axis=0)
frogtraceFlip, tau, w = plot_frog(E=Eflip, t=t, w=w, dt=dt, w0=w0, plot=False)


fig = plt.figure(figsize=(7,7))
gs = fig.add_gridspec(4, 2)
plt.subplots_adjust(wspace=0.2, hspace=0, top=0.95, bottom=0.05, right=0.95, left=0.05)

# plot the original Electric FIeld
ax = fig.add_subplot(gs[0,0])
ax.plot(t, np.real(Eoriginal), color='blue', label='Real $E(t)$')
ax.plot(t, np.imag(Eoriginal), color='red', label='Imag $E(t)$')
ax.legend(loc=1)
ax.text(0.01, 0.9, 'Original $E(t)$', transform=ax.transAxes)
ax.set_title('Ambiguity')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$E(t)$')

# plot the original FROG trace
ax = fig.add_subplot(gs[0,1])
ax.pcolormesh(tau, w, frogtraceOriginal, cmap='jet')
ax.set_title('FROG trace')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$\omega \rightarrow$')


# plot the translate Electric field
ax = fig.add_subplot(gs[1,0])
ax.plot(t, np.real(Etranslated), color='blue')
ax.plot(t, np.imag(Etranslated), color='red')
ax.text(0.01, 0.9, 'Time Shift', transform=ax.transAxes)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$E(t)$')

# FROG trace for translated E
ax = fig.add_subplot(gs[1,1])
ax.pcolormesh(tau, w, frogtraceTranslated, cmap='jet')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$\omega \rightarrow$')

# plot the constant phase shift
ax = fig.add_subplot(gs[2,0])
ax.plot(t, np.real(Econstphase), color='blue')
ax.plot(t, np.imag(Econstphase), color='red')
ax.text(0.01, 0.8, 'Constant\nPhase Shift', transform=ax.transAxes)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$E(t)$')


# const phase frog
ax = fig.add_subplot(gs[2,1])
ax.pcolormesh(tau, w, frogtraceconstphase, cmap='jet')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$\omega \rightarrow$')

# conflip
ax = fig.add_subplot(gs[3,0])
ax.plot(t, np.real(Eflip), color='blue')
ax.plot(t, np.imag(Eflip), color='red')
ax.text(0.01, 0.9, 'Conjugate Flip', transform=ax.transAxes)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$t \rightarrow$')
ax.set_ylabel(r'$E(t)$')


ax = fig.add_subplot(gs[3,1])
ax.pcolormesh(tau, w, frogtraceFlip, cmap='jet')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$\tau \rightarrow$')
ax.set_ylabel(r'$\omega \rightarrow$')

plt.savefig('./ambiguities.png')
plt.show()



