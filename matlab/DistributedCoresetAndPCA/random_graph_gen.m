function G=random_graph_gen(n, p)
%Generate a random graph
% input
% n: number of nodes in the graph
% p: connected probability
% output
% G: adjacent matrix for a connected undirected graph


    G=zeros(n,n);
    while ~isconnected(G)
        G=double(rand(n,n)<p);
        map=(repmat(1:n,n,1)<=repmat((1:n)',1,n));
        G(map)=0;
        G=G+G';
    end
end
