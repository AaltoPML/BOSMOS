import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.axis import Tick
import numpy as np

plt.style.use('seaborn')
colors=['b', 'g', 'r', 'c', 'm', 'y']
tasks = ['DE', 'MR', 'SD', 'RC']


# behavioural fitness for demonstrative example is reduced by 100 to keep the error scale consistent 
ys = [[0.01, 0.01, 0.01, 0.14, 0.33], [0.01, 0, 0, 0.17, 0.33], [1, 0.99, 1., 0.85, 0.5], [10.5, 786.5, 7.65, 3.16, 0], 
      [0.07, 0.23, 0.15, 0.08, 0.33], [0.19, 0.48, 0.27, 0.22, 0.33], [0.85, 0.54, 0.61, 0.82, 0.5], [75.4, 3614.1, 35.56, 20.1, 0],
      [0, 0.21, 0.17, 0, 0.4], [0, 0.45, 0.35, 0, 0.35], [0, 0.25, 0.68, 0, 0.5], [0, 6757.2, 73.41, 0, 0],
      [0.14, 0.21, 0.18, 0.14, 0.44], [0.33, 0.76, 0.29, 0.35, 0.48], [0.47, 0.23, 0.34, 0.35, 0.25], [134, 6698, 88.32, 36, 0] ]

es = [[0.01, 0.00, 0.00, 0.18, 0.3], [0.01, 0, 0, 0.18, 0.23], [1.1, 87.6, 0.20, 0.39, 0],
      [0.06, 0.15, 0.13, 0.06, 0.47], [0.14, 0.4, 0.22, 0.20, 0.20], [9, 273, 3.54, 1.3, 0],
      [0, 0.18, 0.15, 0, 0.49], [0, 0.35, 0.20, 0, 0.2], [0, 399.9, 8.94, 0, 0],
      [0.08, 0.12, 0.11, 0.07, 0.5], [0.24, 0.36, 0.25, 0.22, 0.25], [17.1, 310.2, 5.75, 12.6]]

fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(7., 4.5))
for i in range(0, int(len(axes.flatten())/4) ):
    ax1 = axes[0, i] #.flatten()[i]
    ax2 = axes[1, i] #.flatten()[i+4]
    ax3 = axes[2, i]
    ax4 = axes[3, i]

    if tasks[i] == 'SD':
        xs = [' ', 'MINEBED', 'BOSMOS', '  ', 'Prior']
    else:
        xs = ['ADO', 'MINEBED', 'BOSMOS', 'LBIRD', 'Prior']
    colors=['b', 'g', 'r', 'c', 'm']

    ys1 = reversed(ys[i*4].copy())
    ys2 = reversed(ys[i*4+1].copy())
    ys3 = reversed(ys[i*4+2].copy())
    ys4 = reversed(ys[i*4+3].copy())
    
    es1 = reversed(es[i*3].copy())
    es2 = reversed(es[i*3+1].copy())

    es4 = reversed(es[i*3+2].copy())

    # this locator puts ticks at regular intervals
    if tasks[i] == 'DE':
        loc1 = plticker.MultipleLocator(base=.3)
        loc2 = plticker.MultipleLocator(base=.4)
        loc3 = plticker.MultipleLocator(base=.3)
    else:
        loc1 = plticker.MultipleLocator(base=0.3)
        loc2 = plticker.MultipleLocator(base=.4)
        loc3 = plticker.MultipleLocator(base=.3)
    
    r_xs = reversed(xs.copy())
    r_colors = reversed(colors.copy())
    for x, y, e, c in zip(r_xs, ys1, es1, r_colors):
        m, mw = (None, 0) if ' ' in x else ('o', 1)
        ls = '-' if x == 'BOSMOS' else '--'
        alpha = 1 if x == 'BOSMOS' else 0.6
        eb1 = ax1.errorbar(y, x, xerr=e, fmt=ls, marker=m, markersize=3., capsize=2.5, color=c,  markeredgewidth=mw, alpha=alpha)
        eb1[-1][0].set_linestyle(ls)
    ax1.xaxis.set_major_locator(loc1)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(30)

    # if tasks[i] != 'DE':
    ax1.set_xlim(-0.1, 1.)
    
    r_xs = reversed(xs.copy())
    r_colors = reversed(colors.copy())
    for x, y, e, c in zip(r_xs, ys2, es2, r_colors):
        m, mw = (None, 0) if ' ' in x else ('o', 1)
        ls = '-' if x == 'BOSMOS' else '--'
        alpha = 1 if x == 'BOSMOS' else 0.6
        eb2=ax2.errorbar(y, x, xerr=e, fmt=ls, marker=m, markersize=3., capsize=2.5, color=c,  markeredgewidth=mw, alpha=alpha)
        eb2[-1][0].set_linestyle(ls)
    ax2.xaxis.set_major_locator(loc2)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(30)

    # if tasks[i] != 'DE':
    ax2.set_xlim(0, 1.2)

    r_xs = reversed(xs.copy())
    r_colors = reversed(colors.copy())
    y_pos = np.arange(len(xs))
    for x, y, c in zip(r_xs, ys3, r_colors):
        alpha = 1 if x == 'BOSMOS' else 0.6
        # m, mw = (None, 0) if ' ' in x else ('o', 1)
        ax3.barh(x, y, color=c, alpha=alpha)
    ax3.xaxis.set_major_locator(loc3)
    for tick in ax3.get_xticklabels():
        tick.set_rotation(30)
    ax3.set_xlim(0, 1)
    # ax3.set_yticks(y_pos, labels=r_xs)
    
    #xs = ['ADO', 'MINEBED', 'BOSMOS', 'EIRD']
    #colors = ['b', 'g', 'r', 'c']
    r_xs = reversed(xs.copy())
    r_colors = reversed(colors.copy())
    y_pos = np.arange(len(xs))
    for x, y, c in zip(r_xs, ys4, r_colors):
        if x == 'Prior':
            continue
        alpha = 1 if x == 'BOSMOS' else 0.6
        # m, mw = (None, 0) if ' ' in x else ('o', 1)
        temp = np.array(np.log(y))
        temp[temp==np.NINF] = 0
        ax4.barh(x, temp, color=c, alpha=alpha)
        # print(np.log(y).replace()

    loc4 = plticker.MultipleLocator(base=3)
    ax4.xaxis.set_major_locator(loc4)
    for tick in ax4.get_xticklabels():
        tick.set_rotation(30)
    ax4.set_xlim(0, 10)
    
    
    if tasks[i] != 'DE':
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])

pad = .1 # in points
cols = ['Demonstrative example', 'Memory retention', 'Signal detection', 'Risky choice']
rows = [r"$\eta_{b}$", r"$\eta_{p}$", r"$\eta_{m}$", r"$t_{log}$"]
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad*100),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=10)

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
    
fig.tight_layout(pad=0.5)
plt.savefig('filename.png', dpi=600)



