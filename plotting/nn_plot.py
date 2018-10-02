import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import numpy as np






fig, ax = plt.subplots(1, 1)





for y in [(1/4), (2/4), (3/4)]:
    circle = mpatches.Circle((0.2,y), 0.05, color='red')
    ax.add_patch(circle)

for y in [(3/8), (5/8)]:
    circle = mpatches.Circle((0.7,y), 0.05, color='orange')
    ax.add_patch(circle)

for y in [(3/8), (5/8)]:
    arrow = mpatches.Arrow(0.78, y, 0.1, 0, width=0.1, color='black')
    ax.add_patch(arrow)

for y in [(1/4), (2/4), (3/4)]:
    arrow = mpatches.Arrow(0.02, y, 0.125, 0, width=0.1, color='red')
    ax.add_patch(arrow)


for y in [(1/4), (2/4), (3/4)]:
    for y2, color in zip([(3/8), (5/8)], ['blue', 'green']):
        ax.plot([0.255, 0.645], [y, y2], linewidth=3, color=color)

for y, num in zip(np.array([(1/4), (1.7/4), (2.45/4)])+0.1, ['3', '2', '1']):
    txt = ax.text(0.33, y, r'$w_{'+num+'1}$', color='green')
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

for y, num in zip(np.array([(1.7/4), (2/4), (2.3/4)])-0.1, ['3', '2', '1']):
    txt = ax.text(0.53, y, r'$w_{'+num+'2}$', color='blue')
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

for y, num in zip([(1/4), (2/4), (3/4)], ['3', '2', '1']):

    txt = ax.text(0.19, y, '$x_'+num+'$', color='red')
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

for y, num in zip(np.array([(3/8), (5/8)])+0.04, ['2', '1']):
      ax.text(0.8, y, '$Output_{'+num+'}$')


ax.set_xlim(0, 0.9)
ax.set_ylim(0.18, (1-0.18))

ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
plt.savefig('/home/zom/Documents/MAKETEX_PROJECTS/research_paper/layerdiag.png')


plt.show()