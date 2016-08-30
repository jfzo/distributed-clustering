#!/usr/bin/python

'''
Converts dense CSV file into a sparse matrix file in Cluto's format.

Input file must be a comma separated file that also contains labels in the last field.
'''

import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--outfile", type=str, help="Path to the output file where the sparse matrix will be stored")
parser.add_argument("-i","--inputfile", type=str, help="Path to the input file in CSV format where the dense vectors are stored")
args = parser.parse_args()

if not args.inputfile or not args.outfile:
    print "Usage: ./csv2clutosparse -i <path to the input CSV file> -o <path to output file>"
    sys.exit()

outputfile = args.outfile
inputfile = args.inputfile


outfh = open(outputfile,'w')
infh = open(inputfile)

outfh.write('                     \n') # 20 characters are reserved for the header.
N = 0
for L in infh:
    txt_flds = L.strip().split(',') # it is assumed that only the last field is an int value.
    values = map(float, txt_flds[:-1])
    label  = int(txt_flds[-1])
    D = len(values)
    N += 1
    for v in values:
        
    
infh.close()
outfh.close()








