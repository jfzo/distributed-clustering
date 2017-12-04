import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
import argparse
import sys

from tabulate import tabulate

def print_performance(real_labels, estimated_labels):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(real_labels, estimated_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(real_labels, estimated_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(real_labels, estimated_labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(real_labels, estimated_labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(real_labels, estimated_labels))

def obtain_performance(real_labels, estimated_labels):
    # Homogeneity Completeness  V-measure  Adjusted Rand Index  Adjusted Mutual Information
    return {
        "homogeneity":metrics.homogeneity_score(real_labels, estimated_labels),
        "completeness":metrics.completeness_score(real_labels, estimated_labels),
        "v-score":metrics.v_measure_score(real_labels, estimated_labels), 
        "ari":metrics.adjusted_rand_score(real_labels, estimated_labels),
        "ami":metrics.adjusted_mutual_info_score(real_labels, estimated_labels)
    }



parser = argparse.ArgumentParser(description='Sparse data file name.')
parser.add_argument('-i', type=str, help='file name', required=True)
parser.add_argument('-f', type=str, help='display format (plain, simple, grid, html, latex, latex_raw, latex_booktabs ...). See tabulate package.', required=True)
args = parser.parse_args()

DATA_PATH = args.i

print "Evaluating over corepoints generated from data file", DATA_PATH

#cpdata = np.genfromtxt("{0}.corepoints.csv".format(DATA_PATH), delimiter=' ')
real_lbl = np.loadtxt("{0}.corepoints.labels".format(DATA_PATH),delimiter="\n")
allreal_lbl = np.loadtxt("{0}.labels".format(DATA_PATH),delimiter="\n")
total_num_clusters = np.unique(allreal_lbl).shape[0]


performances = [];
headers = ["Method", "V-Score", "ARI", "AMI", "#Grps(from {0})".format(total_num_clusters)]

"""
db = DBSCAN(eps=0.9, min_samples=10).fit(cpdata)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
#print_performance(real_lbl, labels)
values = obtain_performance(real_lbl, labels)
performances.append(["DBSCAN (scikit)",values["v-score"],values["ari"],values["ami"]])
"""
labels = np.loadtxt("{0}.dsnnfinal.labels".format(DATA_PATH), delimiter="\n")
num_clusters_found = np.unique(labels).shape[0]
#print_performance(real_lbl, labels)
values = obtain_performance(allreal_lbl, labels)
performances.append(["D-SNN (all)",values["v-score"],values["ari"],values["ami"],num_clusters_found])


labels = np.loadtxt("{0}.corepoints.snn.labels".format(DATA_PATH), delimiter="\n")
num_clusters_found = np.unique(labels).shape[0]
#print_performance(real_lbl, labels)
values = obtain_performance(real_lbl, labels)
performances.append(["SNN (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found])

labels = np.loadtxt("{0}.corepoints.conncomps.labels".format(DATA_PATH), delimiter="\n")
num_clusters_found = np.unique(labels).shape[0]
#print_performance(real_lbl, labels)
values = obtain_performance(real_lbl, labels)
performances.append(["Conn. Comps. (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found])

labels = np.loadtxt("{0}.corepoints.cliques.labels".format(DATA_PATH), delimiter="\n")
num_clusters_found = np.unique(labels).shape[0]
#print_performance(real_lbl, labels)
values = obtain_performance(real_lbl, labels)
performances.append(["Max Cliques (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found])

labels = np.loadtxt("{0}.corepoints.lblprop.labels".format(DATA_PATH), delimiter="\n")
num_clusters_found = np.unique(labels).shape[0]
#print_performance(real_lbl, labels)
values = obtain_performance(real_lbl, labels)
performances.append(["Label Prop. (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found])

labels = np.loadtxt("{0}.corepoints.dbscan.labels".format(DATA_PATH), delimiter="\n")
num_clusters_found = np.unique(labels).shape[0]
#print_performance(real_lbl, labels)
values = obtain_performance(real_lbl, labels)
performances.append(["DBSCAN (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found])

#/workspace/cure_large.dat.clustering.7
labels = np.loadtxt("/workspace/cure_large.dat.clustering.7", delimiter="\n")
num_clusters_found = np.unique(labels).shape[0]
#print_performance(real_lbl, labels)
values = obtain_performance(allreal_lbl, labels)
performances.append(["RepBis. (all)",values["v-score"],values["ari"],values["ami"],num_clusters_found])


print tabulate(performances, headers, tablefmt=args.f)
