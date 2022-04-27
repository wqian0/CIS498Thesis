import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle as pk
#output, e_vals = pk.load(open("FIM and evals, hybrid mode, magfield 0_1.pk", "rb"))
F, e_vals, evecs = pk.load(open("FIM_evals_evecs_sec_half.pk", "rb"))
e_vals = np.reshape(e_vals, (1, len(e_vals)))
print(e_vals.shape)
e_vals = [np.sort(w)[::-1] for w in e_vals]
plt.figure(dpi=200)
ax = plt.gca()
ax.set_yscale('log')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Coarsening level')
plt.ylabel('Eigenvalue')
plt.ylim([1e-6, 1e8])
for i in range(len(e_vals)):
    plt.scatter(np.ones_like(e_vals[i]) * (i), e_vals[i], marker="_", s= 150, linewidths=0.2, alpha = 1)
plt.show()