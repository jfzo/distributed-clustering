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
        "ami":metrics.adjusted_mutual_info_score(real_labels, estimated_labels),
        "noise": len(np.where(np.array(estimated_labels) < 0)[0]) / float(len(estimated_labels))
    }



parser = argparse.ArgumentParser(description='Sparse data file name.')
parser.add_argument('-i', type=str, help='file name', required=True)
parser.add_argument('-e', type=str, help='comma separated list of methods to evaluate', required=True)
parser.add_argument('-b', type=str, help='File with benchmark labels', required=False)
parser.add_argument('-f', type=str, help='display format (plain, simple, grid, html, latex, latex_raw, latex_booktabs ...). See tabulate package.', required=True)
args = parser.parse_args()

DATA_PATH = args.i
methods = args.e.split(",")

print "Evaluating over corepoints generated from data file", DATA_PATH


#cpdata = np.genfromtxt("{0}.corepoints.csv".format(DATA_PATH), delimiter=' ')
real_lbl = np.loadtxt("{0}.corepoints.labels".format(DATA_PATH),delimiter="\n")
allreal_lbl = np.loadtxt("{0}.labels".format(DATA_PATH),delimiter="\n")
total_num_clusters = np.unique(allreal_lbl).shape[0]


performances = [];
headers = ["Method", "V-Score", "ARI", "AMI", "#Grps(from {0})".format(total_num_clusters), "%Noise"]


labels = np.loadtxt("{0}.dsnnfinal.labels".format(DATA_PATH), delimiter="\n")
num_clusters_found = np.unique(labels).shape[0]
#print_performance(real_lbl, labels)
values = obtain_performance(allreal_lbl, labels)
performances.append(["D-SNN (all)",values["v-score"],values["ari"],values["ami"],num_clusters_found,values["noise"]])
if args.b:
    #/workspace/cure_large.dat.clustering.7
    labels = np.loadtxt(args.b, delimiter="\n")
    num_clusters_found = np.unique(labels).shape[0]
    #print_performance(real_lbl, labels)
    values = obtain_performance(allreal_lbl, labels)
    performances.append(["Benchmark (all)",values["v-score"],values["ari"],values["ami"],num_clusters_found,values["noise"]])

if "snn" in methods:
    labels = np.loadtxt("{0}.corepoints.snn.labels".format(DATA_PATH), delimiter="\n")
    num_clusters_found = np.unique(labels).shape[0]
    #print_performance(real_lbl, labels)
    values = obtain_performance(real_lbl, labels)
    performances.append(["SNN (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found,values["noise"]])
if "conncomps" in methods:
    labels = np.loadtxt("{0}.corepoints.conncomps.labels".format(DATA_PATH), delimiter="\n")
    num_clusters_found = np.unique(labels).shape[0]
    #print_performance(real_lbl, labels)
    values = obtain_performance(real_lbl, labels)
    performances.append(["Conn. Comps. (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found,values["noise"]])
if "cliques" in methods:
    labels = np.loadtxt("{0}.corepoints.cliques.labels".format(DATA_PATH), delimiter="\n")
    num_clusters_found = np.unique(labels).shape[0]
    #print_performance(real_lbl, labels)
    values = obtain_performance(real_lbl, labels)
    performances.append(["Max Cliques (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found,values["noise"]])
if "lblprop" in methods:
    labels = np.loadtxt("{0}.corepoints.lblprop.labels".format(DATA_PATH), delimiter="\n")
    num_clusters_found = np.unique(labels).shape[0]
    #print_performance(real_lbl, labels)
    values = obtain_performance(real_lbl, labels)
    performances.append(["Label Prop. (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found,values["noise"]])
if "dbscan" in methods:
    labels = np.loadtxt("{0}.corepoints.dbscan.labels".format(DATA_PATH), delimiter="\n")
    num_clusters_found = np.unique(labels).shape[0]
    #print_performance(real_lbl, labels)
    values = obtain_performance(real_lbl, labels)
    performances.append(["DBSCAN (cpts)",values["v-score"],values["ari"],values["ami"],num_clusters_found,values["noise"]])


print tabulate(performances, headers, stralign="right", numalign="left", tablefmt=args.f)
