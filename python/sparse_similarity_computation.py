#!/usr/bin/python
import numpy as np
import numpy as np
import scipy.io as sio
from heapq import *

"""
IndexJoin technique.

Input file must be in Cluto's format. That is, the 1st line contains 3 fields, namely
the number of instances, the dimensionality of the vector space and the number of 
non zero values.
After that, every line contains a ' ' separated list of values, one for the column index 
(starting from 1) and another for the column value.
"""


class InstanceFeature(object):
    """
    This class is used to override the comparison operators of a two value tuple.
    In this way, the heapq can be used as a max queue and not only as a min queue.
    """
    def __init__(self, f_idx, f_val):
        self.f_idx = f_idx
        self.f_val = f_val

    def __eq__(self, other):
        if isinstance(other, InstanceFeature):
            return self.f_val == other.f_val
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, InstanceFeature):
            return self.f_val > other.f_val # inverted in order to allow a max-heap
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, InstanceFeature):
            return self.f_val < other.f_val # inverted in order to allow a max-heap
        return NotImplemented

    def __str__(self):
        return "IX:{0} -> VAL:{1:.4f}\n".format(self.f_idx, self.f_val)


def dense_similarity_from_csv_file(inputfile):
    """
    dense_similarity_from_csv_file(inputfile)
    :param inputfile: Path to the input file in CSV format
    :return: A symmetric instance DOT similarity matrix
    """

    in_fm = open(inputfile)
    doc_ix = 1
    for L in in_fm:
        data_L = L.strip().split()



    in_fm.close()




def sparse_similarity_from_cluto_file(inputfile):
    """
    sparse_similarity_from_cluto_file(inputfile)
    :param inputfile: Path to the input file in cluto's sparse format
    :return: A symmetric instance DOT similarity matrix
    """

    in_fm = open(inputfile)
    N,D,_ = map(int, in_fm.readline().strip().split()) #Number of instances, Number of dimensions and NNZ

    idx = range(D)
    for i in range(D):
        idx[i] = {} # empty dict

    doc_ix = 1
    for L in in_fm:
        data_L = L.strip().split()
        for i in range(0,len(data_L),2):
            f_ix = int(data_L[i])
            f_val = float(data_L[i+1])
            idx[f_ix - 1][doc_ix] = f_val
        doc_ix += 1



    in_fm.seek(0) # reversing file manager
    in_fm.readline() # bypassing header

    S = np.zeros((N,N))
    doc_ix = 1
    for L in in_fm:
        data_L = L.strip().split()
        S[doc_ix-1, doc_ix-1] = 1.0
        for i in range(0,len(data_L),2):
            f_ix = int(data_L[i])
            # check that feature in the index
            for Dc in idx[f_ix - 1]:
                if Dc > doc_ix: # symmetric matrix, hence only upper diagonal values matter.
                    S[doc_ix - 1, Dc - 1] += idx[f_ix - 1][doc_ix] * idx[f_ix - 1][Dc]
                    S[Dc-1, doc_ix-1] = S[doc_ix - 1, Dc - 1]
        doc_ix += 1
    in_fm.close()
    return S


def sparse_similarity_from_mat_file(inputfile):
    """
    sparse_similarity_from_mat_file(inputfile)
    :param inputfile: Path to the input file in Matlab's format
    :return: A symmetric instance DOT similarity matrix
    """

    M = sio.loadmat(inputfile)
    a,b,c,d = M.values()
    if isinstance(a, np.ndarray):
        M = a
    elif isinstance(a, np.ndarray):
        M = b
    elif isinstance(a, np.ndarray):
        M = c
    else:
        M = d
    print "Matrix detected:", M.shape[0], "x", M.shape[1]
    N, D = M.shape

    idx = range(D)
    for i in range(D):
        idx[i] = {} # empty dict

    for doc_ix in range(N):
        for feat_n in range(D):
            f_val = M[doc_ix,feat_n]
            if f_val != 0:
                idx[feat_n][doc_ix] = f_val


    S = np.zeros((N,N))
    for doc_ix in range(N):
        S[doc_ix, doc_ix] = 1.0
        for feat_n in range(D):
            f_val = M[doc_ix,feat_n]
            if f_val != 0:
                #idx[feat_n][doc_ix]
                for Dc in idx[feat_n]:
                    if Dc > doc_ix:
                        S[doc_ix, Dc] += idx[feat_n][doc_ix] * idx[feat_n][Dc]
                        S[Dc, doc_ix] = S[doc_ix, Dc]

    return S


def compute_knn(S, K):
    """
    compute_knn(S, K)
    :param S: Symmetric similarity matrix
    :param K: Number of nearst neighbors to return for each instance
    :return: A list whose i-th position holds a list of the K nearest documents (index and similarity value) to doc i
    """

    N = S.shape[0] # number of instances
    KNN = [None for i in range(N)]

    for i in range(N):
        new_s_i = np.zeros(N)
        h = []
        candidates = np.where(S[0, :] > 0)[0]
        for c in candidates:
            heappush(h, InstanceFeature(c, S[i, c])) # Highest similarities are first
        KNN[i] = [heappop(h) for n in range(K)]  # Trims the Top K similarities nad their corresponding indexes.

    return KNN


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--outfile", type=str, help="Path to the output file where the net will be stored")
    parser.add_argument("-f","--format", type=str, help="Format of the data. Available options: cluto|mat")
    parser.add_argument("-i","--inputfile", type=str, help="Path to the input file in Cluto's format where the sparse vectors are stored")
    args = parser.parse_args()

    if not args.outfile or not args.inputfile or not args.format:
        print "Usage: ./sparse_similarity_computation.py -f cluto|mat -i <path to the input file> -o <path to output file>"
        sys.exit()

    outputfile = args.outfile
    inputfile = args.inputfile


    # for Cluto's format
    if args.format == "cluto":
        print "Reading", inputfile,"..."
        S = sparse_similarity_from_cluto_file(inputfile)
    elif args.format == "mat":
        S = sparse_similarity_from_mat_file(inputfile)
    else:
        print "Error: Format not supported."
        sys.exit()


    # S is a dense symmetric matrix containing the similarity coefficients.
    print "Writing adjacency matrix to CSV file",outputfile

    f = open(outputfile,"w")
    #S.tofile(f, sep=" ", format="%.4f")
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            f.write("{0:.5f} ".format(S[i,j]) )
        f.write("\n")
    f.close()
