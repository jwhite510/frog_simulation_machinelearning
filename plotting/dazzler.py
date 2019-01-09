import numpy as np
import matplotlib.pyplot as plt
import k_diagram
import matplotlib.patches


def draw_rect_from_center(center, width, height):
    # center = (0.5, 0.5)  # x , y
    # width = 0.35
    # height = 0.35
    bottomleft = center[0] - width / 2.0, center[1] - height / 2.0
    rect = matplotlib.patches.Rectangle(bottomleft, width=width, height=height, transform=fig.transFigure,
                                        zorder=0, color='blue')

    return rect







fig = plt.figure()
gs = fig.add_gridspec(3, 3)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.4)


k_diagram.E_t_prop
k_diagram.E_t
k_diagram.f
k_diagram.t
k_diagram.k



# draw incoming laser pulse
ax = fig.add_subplot(gs[1,0])
ax.plot(k_diagram.t, np.real(k_diagram.E_t_prop), color='black')
ax.set_xticks([])
ax.set_yticks([])


# draw outgoing laser pulse
ax = fig.add_subplot(gs[1,2])
ax.plot(k_diagram.t, np.real(k_diagram.E_t), color='black')
ax.set_xticks([])
ax.set_yticks([])


# draw dazzler
ax = fig.add_subplot(gs[1,1])
ax.plot(k_diagram.f[k_diagram.f_crop_l:-k_diagram.f_crop_r], -1*k_diagram.k[k_diagram.f_crop_l:-k_diagram.f_crop_r], color='green')
ax.set_xticks([])
ax.set_yticks([])

# draw a rectangle
rect = draw_rect_from_center(center=(0.5, 0.5), width=0.32, height=0.35)
fig.patches.append(rect)


# draw an arrow
arrow = matplotlib.patches.FancyArrowPatch(
        (0.26, 0.5), (0.35, 0.5), transform=fig.transFigure,
        arrowstyle='simple', mutation_scale=40, alpha=1,
        fc='r'
    )
fig.patches.append(arrow)

# draw an arrow again
arrow = matplotlib.patches.FancyArrowPatch(
        (1 - 0.35, 0.5), (1-0.26, 0.5),  transform=fig.transFigure,
        arrowstyle='simple', mutation_scale=40, alpha=1,
        fc='r'
    )
fig.patches.append(arrow)


# label the dazzler
fig.text(0.5, 0.66, 'AOPDF', ha='center', bbox=dict(facecolor='white',
                                        edgecolor='black',pad=2.0))

# label applied phase
fig.text(0.5, 0.5, '$\phi(\omega)$', color='green', ha='center')



# control computer
rect = draw_rect_from_center(center=(0.5, 0.90), width=0.15, height=0.15)
fig.patches.append(rect)

# arrow from computer
arrow = matplotlib.patches.FancyArrowPatch(
        (0.5, 0.83), (0.5, 0.69), transform=fig.transFigure,
        arrowstyle='simple', mutation_scale=40, alpha=1,
        fc='r'
    )
fig.patches.append(arrow)


# phi label on computer arrow
fig.text(0.56, 0.77, '$\phi(\omega)$', color='green', backgroundcolor='white', ha='center')

# label the computer
fig.text(0.5, 0.87, 'Controller\nComputer', ha='center', backgroundcolor='white', bbox=dict(facecolor='white',
                                        edgecolor='black',pad=2.0))


plt.savefig('./dazzler.png')
plt.show()



