def sparse_mat_from_cluto(inputfile, csv_fname):
    from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
    import numpy as np
    
    in_fm = open(inputfile)
    N,D,_ = map(int, in_fm.readline().strip().split()) #Number of instances, Number of dimensions and NNZ
    
    X = lil_matrix((N, D))
    ln_no = 0
    for L in in_fm:
        inst_fields = L.strip().split(" ")
        for i in range(0, len(inst_fields), 2):
            feat = int(inst_fields[i]) - 1 # cluto starts column indexes at 1
            feat_val = float(inst_fields[i + 1])
            X[ln_no, feat] = feat_val
            
        ln_no += 1
        
    in_fm.close()
    
    assert(ln_no == N)
    
    np.savetxt(csv_fname, X.todense(), delimiter=" ")
    return None
    #return csr_matrix(X), labels


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print "Some parameter is missing."
        print "Usage",sys.argv[0],"input_sparse_fname output_csv_fname"
        print "Example:python ~/clustsparse2csv.py docmat.dat docmat.csv"
        sys.exit(-1)

    sparse_mat_from_cluto(sys.argv[1], sys.argv[2])
