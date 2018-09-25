import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import numpy as np
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)




fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2, top=0.9, bottom=0.1)

y_heights = np.linspace(0.25, 0.75, 6)

# add the circles
for y in y_heights:
    circle = mpatches.Circle((0.2,y), 0.03, color='red')
    ax[0].add_patch(circle)

# add the circles
for y in y_heights[1:-1]:

    circle = mpatches.Circle((0.7,y), 0.03, color='orange')
    ax[0].add_patch(circle)

# add the arrows
for y in y_heights[1:-1]:
    arrow = mpatches.Arrow(0.78, y, 0.1, 0, width=0.05, color='black')
    ax[0].add_patch(arrow)
# add the arrows
for y in y_heights:
    arrow = mpatches.Arrow(0.02, y, 0.125, 0, width=0.05, color='red')
    ax[0].add_patch(arrow)

# add the lines between neurons
for y in y_heights:
    for y2 in y_heights[1:-1]:
        ax[0].plot([0.235, 0.665], [y, y2], linewidth=3, color='black', alpha=0.6)


ax[0].set_xlim(0, 0.9)
ax[0].set_ylim(0.18, (1-0.18))
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_axis_off()


# make plot 2
for y in y_heights:
    circle = mpatches.Circle((0.2,y), 0.03, color='red')
    ax[1].add_patch(circle)

# add the circles
for y in y_heights[1:-1]:

    circle = mpatches.Circle((0.7,y), 0.03, color='orange')
    ax[1].add_patch(circle)

# add the arrows
for y in y_heights[1:-1]:
    arrow = mpatches.Arrow(0.78, y, 0.1, 0, width=0.05, color='black')
    ax[1].add_patch(arrow)

# add the arrows
for y in y_heights:
    arrow = mpatches.Arrow(0.02, y, 0.125, 0, width=0.05, color='red')
    ax[1].add_patch(arrow)

# add lines
dy = y_heights[1] - y_heights[0]
for y in y_heights[1:-1]:
    ax[1].plot([0.235, 0.665], [y, y], linewidth=3, color='black', alpha=0.6)
    ax[1].plot([0.235, 0.665], [y+dy, y], linewidth=3, color='black', alpha=0.6)
    ax[1].plot([0.235, 0.665], [y-dy, y], linewidth=3, color='black', alpha=0.6)



ax[1].set_xlim(0, 0.9)
ax[1].set_ylim(0.18, (1-0.18))

ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_axis_off()






# add letters
for axis, letter in zip([ax[0], ax[1]], ['a)', 'b)']):

    axis.text(-0.1, 1, letter, transform=axis.transAxes)








plt.show()













