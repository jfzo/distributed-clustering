function T=gen_spanning_tree(G)
%Generate a spanning tree for the undirected connected graph G
% first generate a spanning tree by Prim's method
% then pick a random node as the root and convert it to a rooted tree

% input:
% G: the adjacent matrix of a undirected graph
% output:
% T: a spanning tree of the graph G; directed: from parent to
%    children
    
    [Tree, ~] = graphminspantree(sparse(G));% Prim's method
    T0=full(Tree);
    T0=T0'+T0;
    
    visited=-1*ones(1,size(T0,1)); % -1 for not visited; 0 for waiting to span; 1 for already span
    T=zeros(size(T0));
   
    root=randsample(size(T0,1),1);
    visited(root)=0;
    to_span=find(visited==0);
    while ~isempty(to_span)
        for i=1:length(to_span)
            to_span_node=to_span(i);
            children=(T0(to_span_node,:)>0 & visited<0);
            T(to_span_node, children)=1;
            visited(to_span_node)=1;
            visited(children)=0;
        end
        to_span=find(visited==0);
    end
%     visited(root)=0;
%     to_span=find(visited==0);
%     while ~isempty(to_span)
%         to_span=to_span(1);
%         
%         children=(T0(to_span,:)>0 & visited<0);
%         T(to_span, children)=1;
%         
%         visited(to_span)=1;
%         visited(children)=0;
%         to_span=find(visited==0);
%     end
end