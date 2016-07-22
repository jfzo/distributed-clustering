function C = my_intersect(A,B)
    if ~isempty(A)&&~isempty(B)
        P = zeros(1, max(max(A),max(B)) ) ;
        P(A) = 1;
        C = B(logical(P(B)));
    else
        C = [];
    end
    C = unique(C);
end
