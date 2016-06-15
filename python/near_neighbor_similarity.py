#!/usr/bin/python

'''
IndexJoin technique.

Input file must be in Cluto's format. That is, the 1st line contains 3 fields, namely
the number of instances, the dimensionality of the vector space and the number of 
non zero values.
After that, every line contains a ' ' separated list of values, one for the column index 
(starting from 1) and another for the column value.
'''


class InstanceFeature(object):
    '''
    This class is used to override the comparison operators of a two value tuple.
    In this way, the heapq can be used as a max queue and not only as a min queue.
    '''
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




import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-k","--nnbrs", type=int, help="Number of neighbors to consider")
parser.add_argument("-o","--outfile", type=str, help="Path to the output file where the net will be stored")
parser.add_argument("-i","--inputfile", type=str, help="Path to the input file in Cluto's format where the sparse vectors are stored")
args = parser.parse_args()

if not args.nnbrs or not args.outfile or not args.inputfile:
    print "Usage: ./near_neighbor_similarity -k <near neighbor value> -i <path to the input file> -o <path to output file>"
    sys.exit

k = args.nnbrs
outputfile = args.outfile
inputfile = args.inputfile

print "Reading", inputfile,"..."
in_fm = open(inputfile)
N,D,_ = map(int, in_fm.readline().strip().split())

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
import numpy as np

S = np.zeros((N,N))
doc_ix = 1
for L in in_fm:
    data_L = L.strip().split()
    for i in range(0,len(data_L),2):
        f_ix = int(data_L[i])
        # check that feature in the index
        for Dc in idx[f_ix - 1]:
            if Dc > doc_ix: # symmetric matrix, hence only upper diagonal values matter.
                S[doc_ix - 1, Dc - 1] += idx[f_ix - 1][doc_ix] * idx[f_ix - 1][Dc]
            
    doc_ix += 1
in_fm.close()

'''
Filters the similarity matrix by selecting only the highest k values of similarity 
in each row.
'''


from heapq import *

print "Filtering near neighbors (k=",k,")"

for i in range(N):
    new_s_i = np.zeros(N)
    h = []
    candidates = np.where( S[0,:] > 0 )[0]
    for c in candidates:
        heappush(h, InstanceFeature(c,S[i,c]) )
    nnbrs = [heappop(h) for n in range(k)] # list of InstanceFeature objs.
    for n in nnbrs:
        new_s_i[n.f_idx] = n.f_val
    S[i,:] = new_s_i

print "Writing adjacency matrix to",outputfile
f = open(outputfile,"w")
S.tofile(f, sep=" ", format="%.4f")
f.close()