import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.axis import Tick
import numpy as np

plt.style.use('seaborn')
# plt.style.use('tableau-colorblind10')
colors=['blue', 'g', 'red', 'c', 'm', 'y']
tasks = ['DE', 'MR', 'SD', 'RC']


# BEHAVIOURAL FITNESS 
beh_ys = [ [(0.03, 0.02, 0.01), (0.02, 0.01, 0.01), (0.05, 0.01, 0.01)],
           [(0.20, 0.15, 0.07), (0.27, 0.24, 0.23), (0.24, 0.17, 0.15)],
           [(0, 0, 0), (0.27, 0.23, 0.21), (0.25, 0.17, 0.17)],
           [(0.32, 0.27, 0.14), (0.3, 0.26, 0.21), (0.26, 0.24, 0.18)]
       ]
beh_es = [ [(0.03, 0.02, 0.01), (0.07, 0.00, 0.00), (0.07, 0.00, 0.00)],
           [(0.16, 0.10, 0.06), (0.22, 0.19, 0.15), (0.19, 0.14, 0.13)],
           [(0, 0, 0), (0.24, 0.17, 0.18), (0.21, 0.15, 0.15)],
           [(0.11, 0.12, 0.08), (0.11, 0.12, 0.12), (0.11, 0.13, 0.11)]
       ]

# PARAMETER ESTIMATION
par_ys = [ [(0.05, 0.03, 0.01), (0, 0, 0), (0.05, 0, 0)],
           [(0.25, 0.23, 0.19), (0.47, 0.46, 0.48), (0.29, 0.28, 0.27)],
           [(0, 0, 0), (0.60, 0.43, 0.45), (0.37, 0.36, 0.35)],
           [(0.42, 0.43, 0.33), (0.86, 0.81, 0.76), (0.41, 0.45, 0.29)]
        ]

par_es = [ [(0.06, 0.04, 0.01), (0.02, 0, 0), (0.07, 0., 0)],
           [(0.21, 0.19, 0.14), (0.38, 0.39, 0.40), (0.21, 0.20, 0.22)],
           [(0, 0, 0), (0.24, 0.34, 0.35), (0.22, 0.21, 0.20)],
           [(0.22, 0.22, 0.24), (0.28, 0.32, 0.36), (0.25, 0.26, 0.25)]            
        ]

# ABERAGE MODEL ACCURACY
m_ys = [ [(0.97, 0.99, 1), (0.96, 0.99, 0.99), (0.85, 0.99, 1)],
         [(0.66, 0.75, 0.85), (0.61, 0.54, 0.54), (0.63, 0.59, 0.59)],
         [(0, 0, 0), (0.28, 0.27, 0.25), (0.77, 0.7, 0.68)],
         [(0.24, 0.4, 0.47), (0.25, 0.35, 0.23), (0.26, 0.36, 0.34)]
        ]



fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(7., 3.5))
for i in range(0, int(len(axes.flatten())/3) ):
    print(i)
    ax1 = axes[0, i]
    ax2 = axes[1, i]
    ax3 = axes[2, i]

    if tasks[i] == 'SD':
        xs = [' ', 'MINEBED', 'BOSMOS']
    else:
        xs = ['ADO', 'MINEBED', 'BOSMOS']
    colors=['blue', 'g', 'red']

    ys1 = beh_ys[i]
    es1 = beh_es[i]

    ys2 = par_ys[i]
    es2 = par_es[i]

    ys3 = m_ys[i]
    
    # this locator puts ticks at regular intervals
    if tasks[i] == 'DwE':
        loc1 = plticker.MultipleLocator(base=.4)
        loc2 = plticker.MultipleLocator(base=.05)
        loc3 = plticker.MultipleLocator(base=.5)
    else:
        loc1 = plticker.MultipleLocator(base=0.3)
        loc2 = plticker.MultipleLocator(base=.4)
        loc3 = plticker.MultipleLocator(base=.3)

    for j, x, y, e, c in zip(range(len(ys1)), xs, ys1, es1, colors):
        m, mw = (None, 0) if ' ' in x else ('o', 1)
        ls = '-' if x == 'BOSMOS' else '--'
        alpha = 1 if x == 'BOSMOS' else 0.6
        if j == 0 and x == ' ':
            alpha = 0

        temp_x = [1.0 + 0.2*j, 2.06 + 0.14*j, 3.0 + 0.2*j]
        eb1 = ax1.errorbar(temp_x, y, yerr=e, fmt=ls, marker=m, markersize=3., capsize=2.5, color=c,  markeredgewidth=mw, alpha=alpha)
        eb1[-1][0].set_linestyle(ls)
        # ax1.
    ax1.yaxis.set_major_locator(loc1)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(0)
    ax1.set_xticks([1.2, 2.2, 3.2], labels=['', '', ''])
    ax1.set_ylim(-0.1,1.)
    ax1.set_xlim(0.77, 3.63)
    

    for j, x, y, e, c in zip(range(len(ys2)), xs, ys2, es2, colors):
        m, mw = (None, 0) if ' ' in x else ('o', 1)
        ls = '-' if x == 'BOSMOS' else '--'
        alpha = 1 if x == 'BOSMOS' else 0.6
        if j == 0 and x == ' ':
            alpha = 0
        temp_x = [1.0 + 0.2*j, 2.06 + 0.14*j, 3.0 + 0.2*j]
        eb2 = ax2.errorbar(temp_x, y, yerr=e, fmt=ls, marker=m, markersize=3., capsize=2.5, color=c,  markeredgewidth=mw, alpha=alpha)
        eb2[-1][0].set_linestyle(ls)
    ax2.yaxis.set_major_locator(loc2)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(0)
    ax2.set_xticks([1.2, 2.2, 3.2], labels=['', '', ''])
    ax2.set_ylim(-0.1,1.2)
    ax2.set_xlim(0.77, 3.63)

    
    for j, x, y, c in zip(range(len(ys3)), xs, ys3, colors):
        alpha = 1 if x == 'BOSMOS' else 0.6
        # m, mw = (None, 0) if ' ' in x else ('o', 1)
        temp_x = [1 + 0.2*j, 2 + 0.2*j, 3 + 0.2*j]
        ax3.bar(temp_x, y, color=c, alpha=alpha, width=0.2)
    ax3.yaxis.set_major_locator(loc3)
    for tick in ax3.get_xticklabels():
        tick.set_rotation(0)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([1.2, 2.2, 3.2], labels=['1 tr.', '4 tr.', '20 tr.'])
    print(ax3.get_xlim())
    
    if tasks[i] != 'DE':
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
    else:
        ax2.legend(['ADO', 'MINEBED', 'BOSMOS'], loc=2, prop={'size': 8})
        pass

pad = .1 # in points
cols = ['Demonstrative example', 'Memory retention', 'Signal detection', 'Risky choice']
rows = [r"$\eta_{b}$", r"$\eta_{p}$", r"$\eta_{m}$"]
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



