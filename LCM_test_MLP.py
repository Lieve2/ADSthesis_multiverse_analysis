import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

### ----- helper function ----- ###

qfunc = lambda x: abs(norm.ppf(x*(norm.pdf(4) - 0.5) + 0.5))

### ----- begin class maps test ###

# load data
classmap_target0 = pd.read_csv('classmaps/MLP/cm_MLP_netustm.csv')
classmap_target0.rename(columns={'Unnamed: 0': 'observation'}, inplace=True)

# separate classes
classmap_target0_c0 = classmap_target0[classmap_target0['class'] == 0]
classmap_target0_c1 = classmap_target0[classmap_target0['class'] == 1]

# make scatter plot
plt.scatter(y=classmap_target0_c0['prob alternative'],
            x=classmap_target0_c0['farness'],
            c=classmap_target0_c0['colors'])

# add titles to plot
plt.suptitle("Localized class map, class 0")  # change for class 1
plt.title('Model Matthews coefficient: ' + str(classmap_target0_c0['model perf'][0]) + '\n' +
          'Class Matthews coefficient: ' + str(classmap_target0_c0['class perf'][0]),
          fontsize=10)

# x-axis
plt.xlim(-0.05, qfunc(1)+0.01)
plot_probs = np.array([0, 0.5, 0.75, 0.9, 0.99, 0.999, 1])
plt.xticks(qfunc(plot_probs), plot_probs)
plt.xlabel('Localized farness')

# y-axis
plt.ylim(-0.01, 1.05)
plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0])
plt.ylabel("Pr[Alternative class]")

# vertical line
plt.axvline(x=qfunc(0.99), ls=':', color='darkgrey')

# horizontal line
plt.axhline(y=0.5, ls=':', color='darkgrey')

# legend
ptchs = []
for c in range(2):
    ptchs.append(mpatches.Patch(color=["#fd7e14", "#446e9b"][c],
                                label=c))
plt.legend(handles=ptchs,
           loc=0,
           title='Predicted Class')
plt.rcParams["legend.fontsize"] = 6

# show plot
plt.show()
