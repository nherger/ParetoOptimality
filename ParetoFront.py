#######################################################
## Import modules
########################################################
import numpy as np
import matplotlib.pyplot as plt
from Pareto_Frontier import dominates
from Pareto_Frontier import cull
from mpl_toolkits.mplot3d import Axes3D


#######################################################
## Function which finds the 3D Pareto Frontier
########################################################
# Based on http://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def dominates(row, rowCandidate, optimisation):
    if optimisation == 'minimize':
        return all(r < rc for r, rc in zip(row, rowCandidate)) # <: minimize
    elif optimisation == 'maximize':
        return all(r >= rc for r, rc in zip(row, rowCandidate)) # >=: maximize

def cull(pts, dominates, optimisation):
    import numpy as np
    dominated = []
    cleared = []
    remaining = pts
    while remaining:
        candidate = remaining[0]
        new_remaining = []
        for other in remaining[1:]:
            [new_remaining, dominated][dominates(candidate, other, optimisation)].append(other)
        if not any(dominates(other, candidate, optimisation) for other in new_remaining):
            cleared.append(candidate)
        else:
            dominated.append(candidate)
        remaining = new_remaining
    return np.swapaxes(np.array(cleared),0,1)


#####################################################
## Generate random data
#####################################################
ndots = 100
seed = np.random.RandomState(seed=3)
x = np.array([seed.randint(0, 20, ndots).astype(np.float) for i in range(3)]).T  # (100, 3)


#####################################################
## Find Pareto front
#####################################################
pf = np.unique( cull(list(x), dominates, 'minimize') , axis=1)  # (3, 19)


#####################################################
## Visualise Pareto front
#####################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o', c='grey', s=25)
ax.scatter(pf[0, :], pf[1, :], pf[2, :], marker='*',c='red', s=80, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.invert_xaxis()
plt.show()
