function G=grid_graph_gen(n)
%Generate a undirected grid graph of n X n
% input
% n: the grid should contain nXn nodes
% output
% G: adjacent matrix for the undirected grid

    n_node=n*n;
    G=zeros(n_node,n_node);
    for i=1:n
        for j=1:n
            ind=sub2ind([n n],i,j);
            if (i+1<=n)
                ind2=sub2ind([n n],i+1,j);
                G(ind,ind2)=1;
            end
            if (j+1<=n)
                ind2=sub2ind([n n],i,j+1);
                G(ind,ind2)=1;
            end
        end
    end
    G=G+G';
end