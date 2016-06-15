path('../DistributedCoresetAndPCA', path())
%%
clear;
P=importdata('../data/vary-density.csv');
Pwclass = P;
Z = zeros(size(Pwclass));
Z(1:50,:) = Pwclass(Pwclass(:,3)==3,:);
Z(51:100,1) = Pwclass(Pwclass(:,3)==2,1)-0.2;
Z(51:100,2) = Pwclass(Pwclass(:,3)==2,2)-0.2;
Z(51:100,3) = zeros(50,1)+1;
Z(101:end,:) = Pwclass(Pwclass(:,3)==1,:);
gscatter(Z(:,1),Z(:,2),Z(:,3))
Pwclass = Z;
P = Z(:,1:2);
%%
K = length(unique(Pwclass(:,3))); % it can be manually set also!    
A = pdist2(P,P);
sigma = 1;
A = exp(-(A.^2)/(2*sigma^2));
for i=1:size(A,1)   
    A(i,i ) = 0;
end

D = sum(A, 2);
D_A = diag((D.^-1).^(1/2));
L = D_A * A * D_A;
[X, eVal] = eigs(L, K);

Y = diag(1./(sum(X.^2,2).^(1/2)) ) * X;
[centers]=lloyd_kmeans(K, Y);

dims = size(Y);
labeledS = zeros(dims(1), 1);

for i=1:dims(1)
    min_d = inf;
    min_c = -1;
    for c=1:size(centers,1)
        curr_d = sum((Y(i,:)-centers(c,:)).^2);
        if curr_d < min_d
            min_d = curr_d;
            min_c = c;
        end
    end
    labeledS(i,1) = min_c;
end

%%
gscatter(P(:,1),P(:,2),ones(size(P,1)),'m')
gscatter(Y(:,1),Y(:,2),labeledS(:,1))

figure1 = figure('PaperSize',[20.98404194812 29.67743169791]);
axes1 = axes('Parent',figure1);
hold(axes1,'all');

line(P(Pwclass(:,3)==1,1),P(Pwclass(:,3)==1,2),'Parent',axes1,'Marker','square','LineStyle','none',...
'Color',[1 0 0],'DisplayName','Orig.Data, class +');

line(P(Pwclass(:,3)==2,1),P(Pwclass(:,3)==2,2),'Parent',axes1,'MarkerSize',8,'Marker','v',...
'LineStyle','none','Color',[0 1 1],'DisplayName','Orig.Data, class -');
legend(axes1,'show');

line(P(Pwclass(:,3)==3,1),P(Pwclass(:,3)==3,2),'Parent',axes1,'MarkerSize',8,'Marker','o',...
'LineStyle','none','Color',[0 0 1],'DisplayName','Orig.Data, class -');
legend(axes1,'show');

gscatter(P(:,1),P(:,2),labeledS(:,1))