module DSNN_Similarity

    function cosine(X::SparseMatrixCSC{Float64, Int64})
        #=
        X : Sparse matrix having the objects in its columns and the features in its rows.
        It is assumed that every column is a unit-norm vector.
        =#
        return transpose(X) * X;
    end


end
