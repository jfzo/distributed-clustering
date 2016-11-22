from sparse_similarity_computation import compute_knn
import numpy as np


def compute_snn(KNN):
    """
    compute_SNN(KNN)
    :param KNN:A list whose i-th position holds a list of the K nearest documents (index and similarity value) to instance i
    :return:A symmetric Shared-Nearest-Neighbors similarity matrix
    """

    N = len(KNN)
    S = np.zeros((N,N))

    for i in range(N-1):
        #print "First KNN[",i ,"] ",KNN[i][0].f_idx," / ",KNN[i][0].f_val
        nnbrs_i = {data_tuple.f_idx for data_tuple in KNN[i]}
        for j in range(i+1, N):
            nnbrs_j = {data_tuple.f_idx for data_tuple in KNN[j]}
            S[i,j] = len(nnbrs_i & nnbrs_j)
            S[j, i] = S[i, j]
    return S


def snn_clustering(S, K, Eps, MinPts):
    """
    SNNClustering(S, K, Eps, MinPts)
    :param S: Symmetric similarity matrix
    :param K: Number of nearest neighbors to use
    :param Eps:
    :param MinPts:
    :return: The core point list and the clustetr assignment for each point in the Data.
    """

    knn_info = compute_knn(S, K)
    snn_sim = compute_snn(knn_info)

    N = snn_sim.shape[0]
    # Computing Density ~ #Close-points (in terms of SNN similarity) for each point

    snn_density = np.zeros( (N, ) ) # array to store density
    for i in range(N):
        snn_density[i] = len(np.where(snn_sim[i, :] >= Eps)[0]) # density of instance i

    # Identifying core points (those having snn-density higher than MinPts )
    core_points = np.where(snn_density >= MinPts)[0]

    print "#core_points:",len(core_points)," min-density:",np.min(snn_density)," max-density:",np.max(snn_density)
    assert(len(core_points) > 0 )

    cluster_assign = np.zeros((N,))
    curr_cluster = 1
    n_corepts = len(core_points)
    for core_i in range(n_corepts):
        if cluster_assign[core_points[core_i]] == 0:
            cluster_assign[core_points[core_i]] = curr_cluster
            curr_cluster += 1
        for core_j in range(core_i + 1, n_corepts):
            if snn_sim[core_points[core_i], core_points[core_j]] >= Eps:
                cluster_assign[core_points[core_j]] = cluster_assign[ core_points[core_i] ]

    # Marking noise points with a -1 in the cluster assignment
    non_core_points = np.where(snn_density < MinPts)[0]


    for noncore_pt in non_core_points:
        # find its nearest corepoint
        max_snn_pt = core_points[0]
        for core_pt in core_points:
            if snn_sim[noncore_pt, core_pt] > snn_sim[noncore_pt, max_snn_pt]:
                max_snn_pt = core_pt
        if snn_sim[noncore_pt, max_snn_pt] < Eps: # not within a radius of Eps to its nearest core-point
            cluster_assign[noncore_pt] = -1
        else:
            cluster_assign[noncore_pt] = cluster_assign[max_snn_pt]

    return core_points, cluster_assign


# Example
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
DATA[0:30,:] = np.random.randn(30,2).dot(np.array([1,0,0,1]).reshape((2,2))) + np.array([0,1])
DATA[30:80,:] = np.random.randn(50,2).dot(np.array([1,0,0,1]).reshape((2,2))) + np.array([5,4])
DATA[80:100,:] = np.random.randn(20,2).dot(np.array([1,0.3,0.3,1]).reshape((2,2))) + np.array([0,5])

plt.plot(DATA[:,0], DATA[:,1], 'x');plt.show()

D = euclidean_distances(DATA, DATA)
S = np.max(D) - D
plt.imshow(S)
plt.show()
"""


"""
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
import pylab

noisy_circles = datasets.make_circles(n_samples=100, factor=.5,
                                      noise=.05)
DATA, y = noisy_circles
DATA = StandardScaler().fit_transform(DATA)

D = euclidean_distances(DATA, DATA)
S = np.max(D) - D

CP, CL = snn_clustering(S, 5, 1, 9)

# data with the core-points marked
pylab.scatter(DATA[:,0], DATA[:,1],marker='x');pylab.scatter(DATA[CP,0], DATA[CP,1]);pylab.show()

# data with labels found
pylab.scatter(DATA[np.where(CL != -1),0], DATA[np.where(CL != -1),1],c =CL[np.where(CL != -1)]);pylab.show()
"""
