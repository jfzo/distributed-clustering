function G=random_graph_gen_by_pref_attach(n, m0, m)
%Generate a random graph by the preference attachment-style rule

% input
% n: number of nodes in the graph
% m0: initial connected nodes; >=2, <=n
% optinal:
% m: number of edges of newly added nodes; default to be m0
% output
% G: adjacent matrix for a connected undirected graph

    if nargin<3
        m=m0;
    end
    
    G=zeros(n,n);
    for i=1:m0
        for j=1:(i-1)
            G(i,j)=1;
        end
    end
    degree=zeros(n,1);
    degree(1:m0)=m0-1;
    for i=(m0+1):n
       for j=1:m
           attach_to=randsample(i-1, 1, true, degree(1:(i-1)));
           if G(i,attach_to)~=1
               G(i,attach_to)=1;
               degree(attach_to)=degree(attach_to)+1;
               degree(i)=degree(i)+1;
           end
       end
    end
    G=G+G';
end