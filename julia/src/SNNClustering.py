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


def snn_clustering(snn_sim, Eps, MinPts):
    """
    SNNClustering(snn_sim, Eps, MinPts)
    :param snn_sim: Symmetric Shared-Nearest-Neighbor similarity matrix
    :param Eps:
    :param MinPts:
    :return: The core point, non-core point and noise point lists along with the cluster assignment for each data instance
    """

    N = snn_sim.shape[0]
    # Computing Density ~ #Close-points (in terms of SNN similarity) for each point

    snn_density = np.zeros( (N, ) ) # array to store density
    for i in range(N):
        snn_density[i] = len(np.where(snn_sim[i, :] >= Eps)[0]) # density of instance i

    # Identifying core points (those having snn-density higher than MinPts )
    core_points = np.where(snn_density >= MinPts)[0]

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

    noise_points = np.where(cluster_assign[non_core_points] == -1)[0]
    non_core_points = np.where( cluster_assign[non_core_points] != -1 )[0]


    #print "#core_points:", len(core_points), "#non_core_points:",len(non_core_points),"#noisy_points:",len(noise_points), "min-density:", np.min(snn_density), " max-density:", np.max(snn_density)

    return core_points, non_core_points, noise_points, cluster_assign


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



if __name__ == '__main__':
    import numpy as np
    from sklearn import cluster, datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import euclidean_distances
    import pylab
    import sys
    from clustering_scores import clustering_scores

    #sys.path.append('/Volumes/SSDII/Users/juan/git/distributed-clustering/python/')

    noisy_circles = datasets.make_circles(n_samples=1000, factor=.1, noise=.05)
    DATA, y = noisy_circles
    DATA = StandardScaler().fit_transform(DATA)

    D = euclidean_distances(DATA, DATA)
    S = 1 - (D - np.min(D))/(np.max(D)-np.min(D))

    for K in [90, 100, 110, 120]:
        knn_info = compute_knn(S, K) # sparsify the similarity matrix
        snn_sim = compute_snn(knn_info) # obtains the snn similarity matrix

        max_vm, max_vm_params = 0, [0,0,K]
        for Eps in [5, 10, 15,20,25,30,35, 40]:
            for MinPts in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
                try:
                    CP, NCP, NP, CL = snn_clustering(snn_sim, Eps, MinPts) #corepoints, non-corepoints, noise-points and cluster assignment
                    print "#core_points:", len(CP), "#non_core_points:", len(NCP), "#noisy_points:", len(NP), "#Clusters:", len(np.unique(CL[CP]) )
                    results = clustering_scores(y, CL, display=False) # dict containing {'E','P','ARI','AMI','NMI','H','C','VM'}

                    if results['VM'] > max_vm:
                        max_vm = results['VM']
                        max_vm_params = Eps, MinPts, K

                    print "Eps",Eps,"MinPts", MinPts,"K:",K,"-- VM:",results["VM"],"(",max_vm,")"
                except AssertionError:
                    print("Uups!  No se encontraron Core-Points.  Intentando de nuevo...")


    # BEST CONFIGURATION
    print "Best VM(",max_vm,") is achieved with","Eps:",max_vm_params[0],"MinPts:",max_vm_params[1],"K:",max_vm_params[2]
    knn_info = compute_knn(S, max_vm_params[2])  # sparsify the similarity matrix
    snn_sim = compute_snn(knn_info)  # obtains the snn similarity matrix
    CP, NCP, NP, CL = snn_clustering(snn_sim, max_vm_params[0], max_vm_params[1])
    print "#core_points:", len(CP), "#non_core_points:", len(NCP), "#noisy_points:", len(NP), "#Clusters:", len(np.unique(CL[CP]) )

    # data with the core-points marked
    pylab.subplot(211)
    pylab.scatter(DATA[NCP,0], DATA[NCP,1],marker='^', color='b', label='non-core-point')
    pylab.scatter(DATA[NP,0], DATA[NP,1], marker='+', color='r', label='noise')
    pylab.scatter(DATA[CP,0], DATA[CP,1], marker='o', color='g', label='core-point')
    pylab.legend(loc='upper right')
    # data with labels found
    pylab.subplot(212)
    pylab.scatter(DATA[NCP,0], DATA[NCP,1], c =CL[NCP], marker='o')
    pylab.scatter(DATA[CP, 0], DATA[CP, 1], c=CL[CP], marker='o')
    pylab.scatter(DATA[NP, 0], DATA[NP, 1], marker='x', color='r')

    pylab.show()

