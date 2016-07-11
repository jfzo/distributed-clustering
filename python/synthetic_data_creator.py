print(__doc__)

import time

import numpy as np

import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import pandas as pd

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 10000

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)


colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)


X, y = noisy_circles
X = StandardScaler().fit_transform(X)
df = pd.DataFrame({'A' : X[:,0] , 'B' : X[:,1], 'C' : y})
df.to_csv("noisy_circles.csv", header=False, index=False)

plt.scatter(X[:,0], X[:,1], color=colors[y].tolist(), s=10)
plt.show()


X, y = noisy_moons
X = StandardScaler().fit_transform(X)
df = pd.DataFrame({'A' : X[:,0] , 'B' : X[:,1], 'C' : y})
df.to_csv("noisy_moons.csv", header=False, index=False)
plt.scatter(X[:,0], X[:,1], color=colors[y].tolist(), s=10)
plt.show()
