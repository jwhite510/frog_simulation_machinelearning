import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate



class Field():

    def __init__(self, f0, tau, tmax, tshift, N, ir_scale=None):

        dt = (2 * tmax) / N
        t = dt * np.arange(-N / 2, N / 2)
        Et = np.exp(-2 * np.log(2) * t ** 2 / tau ** 2) * np.exp(1j * 2 * np.pi * f0 * t)
        t = t - tshift

        self.Et = Et
        self.t = t

        if ir_scale:
            self.Et = self.Et * ir_scale





def plot_long_ir():

    # define IR
    ir = Field(f0=0.5e14, tau=45e-15, tmax=80e-15, tshift=0, N=256)

    xuv_list = []
    # define XUV
    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=-5.34e-15, N=256, ir_scale=0.7))
    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=5.34e-15, N=256, ir_scale=0.7))

    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=15e-15, N=256, ir_scale=0.5))
    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=-15e-15, N=256, ir_scale=0.5))

    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=25e-15, N=256, ir_scale=0.25))
    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=-25e-15, N=256, ir_scale=0.25))

    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=35e-15, N=256, ir_scale=0.17))
    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=-35e-15, N=256, ir_scale=0.17))

    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(2,2)
    ax = fig.add_subplot(gs[:,:])

    ax.plot(ir.t, np.real(ir.Et), color='red', label='Infrared Driving Laser')
    ax.plot(xuv_list[0].t, np.real(xuv_list[0].Et), color='blue', label='XUV Generation', alpha=0.6)
    for xuv in xuv_list[1:]:
        ax.plot(xuv.t, np.real(xuv.Et), color='blue', alpha=0.6)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-70e-15, 70e-15)
    ax.set_xlabel(r'time $\rightarrow$')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc=4)
    ax.set_title("Long IR pulse duration")
    plt.savefig("./longduration.png")


def plot_short_ir():
    # define IR
    ir = Field(f0=0.5e14, tau=20e-15, tmax=80e-15, tshift=0, N=256, ir_scale=1.5)

    xuv_list = []
    # define XUV
    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=-5.34e-15, N=256, ir_scale=1.2))
    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=5.34e-15, N=256, ir_scale=1.2))

    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=15e-15, N=256, ir_scale=0.17))
    xuv_list.append(Field(f0=8e14, tau=1.5e-15, tmax=20e-15, tshift=-15e-15, N=256, ir_scale=0.17))

    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, :])

    ax.plot(ir.t, np.real(ir.Et), color='red', label='Infrared Driving Laser')
    ax.plot(xuv_list[0].t, np.real(xuv_list[0].Et), color='blue', label='XUV Generation', alpha=0.6)
    for xuv in xuv_list[1:]:
        ax.plot(xuv.t, np.real(xuv.Et), color='blue', alpha=0.6)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-70e-15, 70e-15)
    ax.set_xlabel(r'time $\rightarrow$')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc=4)
    ax.set_title("Shortened IR pulse duration")
    plt.savefig("./shortduration.png")





plot_long_ir()
plot_short_ir()


plt.show()






