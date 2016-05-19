This file gives a list of functions and a brief description of their functionality with respect to the distributed PCA, coreset construction and k-means clustering algorithms implementation.

Following functions provide the implementation of the main algorithms used.

1. Distributed PCA:

a. distributed_pca.m: This file contains principal component analysis algorithm implementation for distributed data set. Projection of the data set into a lower dimension space is returned, which can be used for reducing dimension of high-dimension data set.

  
2. Coreset Construction:

a. coreset.m: Contains codes for building coreset on weighted/unweighted data points with pre-computed constant approximation solution. Returns an array of coreset points with the respective weights of the points.

b. combined_coreset.m: Contains implementation for the COMBINED coreset algorithm for distributed data set. It simply returns the coreset points by combining all the coresets of partitioned data. 

c. distributed_coreset.m: Contains implementation for the distributed coreset construction algorithm, which takes into account the local coreset cost of each node in network while selecting points to form global coreset. 

d. merge_coreset_on_tree.m: Contains coreset implementation of the path-aware algorithm in the paper "Approximate Clustering on Distributed Data Streams" 


3. k-means Clustering:

a. lloyd_kmeans: Contains implementation of Lloyd¡¯s method for k-means, using kmeans++ for seeding. Returns the k-means cluster centers along with the clustering cost.

b. kmeanspp.m: Contains code to compute seed centers for the first iteration of the Lloyd¡¯s k-means implementation using kmeans++ algorithm.

c. single_swap.m: Contains the single swap algorithm for k-means from the paper "A Local Search Approximation Algorithm for k-Means Clustering".


4. Graph and Partition Functions:

a. get_partition.m: Contains different partition methods for distributing data manually among the nodes of the network. It returns an array of indices denoting the node to which the data point is assigned.

b. grid_graph_gen.m: Contains code to generate a undirected grid graph of n X n

c. random_graph_gen.m: Contains code to generate a random graph of node connectivity given by probability p

d. gen_spanning_tree.m: Contains code to generate a spanning tree for the undirected connected graph G

e. random_graph_gen_by_pref_attach.m: Contains code to generate a random graph by the preferential attachment-style rule

f. isconnected.m: Contains code to determine if a graph is connected

5. Other functions used:

a. sqDistance.m: Contains code to compute the square distances between data points in X & Y

b. sqDistanceIn.m: Contains code to compute the square distances between data points in X

c. consistify_sample_weight.m: Contains code to make sure the weights are non-negative values with at least one positive value so that it can be used in the function randsample


Example:

example.m: The example file contains a mock up of the usage of all the functions given above to construct a k-means clustering solution for an example data set, Pendigits. We first project it to lower dimension using distributed PCA, then use one of the distributed coreset construction algorithms, to build the coreset. Lloyd¡¯s kmeans method on coreset data points gives us a good approximation of the solution at low communication cost.
