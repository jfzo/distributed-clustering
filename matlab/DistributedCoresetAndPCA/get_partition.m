function indn=get_partition(partition,N,n,degree,columns,D)
%partition global data into local data sets
% input:
% partition: partition method, one of 'uniform' 'unbalanced' 'cluster'
%       'weighted' 'degree' 'range'
% N: number of data points in global data set
% n: number of local data sets
% degree: degrees of the local nodes in the communication graph; only used
%       for degree-based partition method
% columns: no. of columns to be considered for range partitioning
% D: data matrix, each row is a data point; only used for similarity-based
%       partition method
% output:
% indn: an array of indices; indn(i) is the index of the node that data
%       point i is assigned to

    indn=ones(N,1);
    if strcmp(partition,'uniform') % uniform partition
        indn=uniform_partition(N,n);
    elseif strcmp(partition,'unbalanced') % unbalanced partition
        indn=unbalanced_partition(N,n);
    elseif strcmp(partition,'cluster') % similarity-based partition
        indn=cluster_partition(D,n);
    elseif strcmp(partition,'weighted') % weighted partition
        indn=weighted_partition(N,n);
    elseif strcmp(partition,'degree') % degree-based partition
        indn=degree_partition(N,n,degree);
    elseif strcmp(partition,'range') % range partition
        indn=range_partition(D,N,n,columns);
    else
        fprintf('Error: do not support partition type %s\n', partition);
    end
end

function indn=uniform_partition(N,n)
    indn=ceil(rand(N,1)*n);
end

function indn=unbalanced_partition(N,n)
    n_large=floor(n/2); % number of nodes that have large size, all the other nodes have small size
    ratio=20; % ratio between the large and small size
    
    % partition
	large_range=ratio*n_large;
	small_range=n-n_large;
    tmp_indn=ceil(rand(N,1)*(large_range+small_range));
	indn=ones(N,1);
    for i=1:N
        if tmp_indn(i)<=large_range
			indn(i)=ceil(tmp_indn(i)/ratio);
		else
			indn(i)=ceil(tmp_indn(i)-large_range+n_large);
        end
    end
end

function indn=cluster_partition(D,n)
	[N,~]=size(D);
	
	p_node=randsample(N,n);
	indn=ones(N,1);
	Dist=sqDistance(D(p_node,:),D);
	for i=1:N
        prob=exp(-1*Dist(:,i)/mean(Dist(:,i)));
		indn(i)=randsample(n,1,true, prob);
	end
end

function indn=weighted_partition(N,n)	
	indn=ones(N,1);
    w=abs(randn(n,1));
    for i=1:N
        indn(i)=randsample(n,1,true, w);
    end
    
end

function indn=degree_partition(N,n,degree)	
	indn=ones(N,1);
	for i=1:N
		indn(i)=randsample(n,1,true, degree);
	end
end

function indn=range_partition(D,N,n,column)
    %first partition on virtual processors to avoid skewed distribution
    n_virt=n*20;
    [f,x] = hist(D(:,column),n_virt);
    cum_f = zeros(1,n_virt);
    cum_f(1,1) = f(1,1);
    for i=2:n_virt
        cum_f(1,i) = cum_f(1,i-1)+f(1,i);
    end
    delta = x(1,2)-x(1,1);
    k = zeros(1,n+1);
    for i=2:n
        k(1,i) = (i-1)*(N/n);
        for j=1:n_virt
            if(k(1,i)<cum_f(1,j));
                break;
            end    
        end
        
        if(j>1)
            k(1,i) = x(1,j) + delta*((k(1,i)-cum_f(1,j-1))/f(1,j));
        else
            k(1,i) = x(1,j) + delta*(k(1,i)/f(1,j));
        end
    end
    k(1,1) = min(D(:,column));
    k(1,n+1) = max(D(:,column));
    indn = zeros(N,1);
    for j= 1:N       
        for i=1:n
          if((k(1,i)<=D(j, column)) & (D(j,column)<=k(1,i+1)))
              indn(j,1) = i;
              break;
          end
        end
    end
end

% function indn = unbalanced_range_partition(D,N,n,columns)
%     min_D = min(D);
%     range_D = range(D);
%     offset = range_D(1, 1:columns)/n;
%     ranges = zeros(n, columns);
%     indn = zeros(N,1);
%     for i = 1:n
%         ranges(i,1:columns) = min_D(1, 1:columns) + i*offset; 
%         for j= 1:N       
%             if((min_D(1,1:columns)<=D(j, 1:columns)) & (D(j, 1:columns)<=ranges(i,1:columns)))
%                 if(indn(j,1)==0)
%                    indn(j,1) = i;
%                 end
%             end    
%         end
%     end
%     indn(indn==0) = n;
% end
% 
% function indn= similarity_balanced_partition(D,N,n,columns)
%     capacity = ceil(N/n);
%     [N,~]=size(D);
%     count= zeros(n,1);
% 	
% 	p_node=randsample(N,n);
% 	indn=ones(N,1);
% 	Dist=sqDistance(D(p_node,:),D);
%     [Dist_min,I_min] = min(Dist);
%     for i=1:N
%         index = check_capacity_assign(I_min(1,i), capacity, n, count);
%         indn(i,1) = index;
%         count(index,1) = count(index,1)+1;
%     end
% end
% 
% function free_index = check_capacity_assign(I_min, capacity, n, count)
%     free_index = I_min;
%     while(1)
%         if(count(free_index,1)<capacity)
%             break;
%         else
%             free_index = free_index+1;
%         end    
%     end
% end


