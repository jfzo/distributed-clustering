module SNNGraphUtil

    function sorted_list_adjacent_nodes(S::Array{Float64,2}, v::Int64)
        adj_l = find(x->x>0, S[v,:]);
        return adj_l[ sortperm(S[v, adj_l], rev=true) ]
    end

    function list_adjacent_nodes(S::Array{Float64,2}, v::Int64)
        adj_l = find(x->x>0, S[v,:]);
        return adj_l
    end


    function nearest_cp_bfs(S::Array{Float64,2}, start::Int64; target::Int64=-1)
        N = size(S,1);
        visited = fill(false, N);
        parent = fill(-1, N);
        weights = fill(Inf, N);
        output = Int64[];
        q = Queue(Int64);    

        enqueue!(q, start)
        visited[start] = true;
        weights[start] = 0;
        push!(output, start)

        while ~isempty(q)
            v = dequeue!(q)
            adj_vlist = sorted_list_adjacent_nodes(S, v)

            for n=collect(1:length(adj_vlist))
                if visited[adj_vlist[n]]
                    continue
                end
                enqueue!(q, adj_vlist[n])
                visited[adj_vlist[n]] = true;
                push!(output, adj_vlist[n])
                weights[adj_vlist[n]] = weights[v] + 1;#S[v,adj_vlist[n]];
                parent[adj_vlist[n]] = v;

                if adj_vlist[n] == target
                    while ~isempty(q)
                        dequeue!(q)
                    end
                    break
                end
            end
        end

        return Dict("paths"=>output, "weights"=>weights, "parents"=>parent)
    end

    function shortest_path(D::Array{Float64,2}, start::Int64; target::Int64=-1)
        N = size(D,1);
        visited = fill(false, N);
        prev = fill(-1, N);    
        dist = fill(Inf, N);
        unvisited = Set(collect(1:N))

        dist[start] = 0;


        while ~isempty(unvisited)
            unvisited_list = collect(unvisited);
            currV = unvisited_list[indmin(dist[unvisited_list])]

            neighbors = list_adjacent_nodes(D, currV)

            for i=collect(1:length(neighbors))
                if visited[neighbors[i]]
                    continue
                end
                alt_dist = dist[currV] + D[currV, neighbors[i]];
                if alt_dist < dist[neighbors[i]]
                    dist[neighbors[i]] = alt_dist;
                    prev[neighbors[i]] = currV;
                end            
            end
            visited[currV] = true;
            delete!(unvisited, currV);

        end

        return Dict("dist"=>dist, "prev"=>prev)
    end

    function add_edge(S::Array{Float64,2}, n1::Int64, n2::Int64; weight::Float64=1.0)
        N = size(S,1)
        assert(n1 <= N && n2 <= N)
        S[n1,n2] = weight;
        S[n2,n1] = weight;
    end

    function add_edge(S::Array{Float64,2}, n1::Int64, n2::Int64, weight::Int64)
        add_edge(S, n1, n2, weight=Float64(weight))
    end

    function bfs_demo1()
        S_demo = zeros(9,9);
        a,b,c,d,e,f,g,h,s = 1,2,3,4,5,6,7,8,9
        add_edge(S_demo, a,b);
        add_edge(S_demo, a,s);
        add_edge(S_demo, s,c);
        add_edge(S_demo, s,g);
        add_edge(S_demo, c,d);
        add_edge(S_demo, c,e);
        add_edge(S_demo, c,f);
        add_edge(S_demo, g,f);
        add_edge(S_demo, g,h);
        add_edge(S_demo, e,h);

        res = nearest_cp_bfs(S_demo, a);
        println("Distances to start vertex:",res["weights"])
        println("Paths from start vertex:",res["paths"])
        #a b s c g d e f h  -> 1,2,9,3,7,4,5,6,8
        #Distances to start vertex:[0.0,1.0,2.0,3.0,3.0,3.0,2.0,3.0,1.0]
        #Paths from start vertex:[1,2,9,3,7,4,5,6,8]
    end


    function bfs_demo2()
        S_demo = zeros(9,9);
        add_edge(S_demo, 1,2, 5);
        add_edge(S_demo, 1,3, 3);
        add_edge(S_demo, 2,3, 3);
        add_edge(S_demo, 3,4, 2);
        add_edge(S_demo, 3,5, 3);
        add_edge(S_demo, 4,5, 1);
        add_edge(S_demo, 4,6, 4);
        add_edge(S_demo, 4,7, 5);
        add_edge(S_demo, 6,7, 2);
        add_edge(S_demo, 6,8, 3);
        add_edge(S_demo, 6,9, 2);
        add_edge(S_demo, 8,9, 4);

        res = nearest_cp_bfs(S_demo, 1);
        println("Weights from start vertex:",res["weights"])
        println("Paths from start vertex:",res["paths"])
        println("Parents:",res["parents"])
        #a b s c g d e f h  -> 1,2,9,3,7,4,5,6,8
        #Distances to start vertex:[0.0,1.0,2.0,3.0,3.0,3.0,2.0,3.0,1.0]
        #Paths from start vertex:[1,2,9,3,7,4,5,6,8]
    end


    function bfs_demo3()
        S_demo = zeros(10,10);
        add_edge(S_demo, 1,9);
        add_edge(S_demo, 2,3);
        add_edge(S_demo, 2,4);
        add_edge(S_demo, 2,8);
        add_edge(S_demo, 2,10);
        add_edge(S_demo, 3,5);
        add_edge(S_demo, 3,9);
        add_edge(S_demo, 4,5);
        add_edge(S_demo, 4,6);
        add_edge(S_demo, 6,7);
        add_edge(S_demo, 7,8);
        add_edge(S_demo, 9,10);

        res = nearest_cp_bfs(S_demo, 3);
        println("Weights from start vertex:",res["weights"])
        println("Paths from start vertex:",res["paths"])
        println("Parents:",res["parents"])
        #a b s c g d e f h  -> 1,2,9,3,7,4,5,6,8
        #Distances to start vertex:[0.0,1.0,2.0,3.0,3.0,3.0,2.0,3.0,1.0]
        #Paths from start vertex:[1,2,9,3,7,4,5,6,8]
        w = 6;
        lPath = Int64[];
        currentV = w;
        while res["parents"][currentV] != -1
            push!(lPath, currentV)
            currentV = res["parents"][currentV];
        end
        push!(lPath, 3);
        println("Shortestpath:",lPath)
    end


    function sp_demo()
        S_demo = zeros(8,8);
        add_edge(S_demo, 1,2,8);
        add_edge(S_demo, 1,3,2);
        add_edge(S_demo, 1,4,5);
        add_edge(S_demo, 2,4,2);
        add_edge(S_demo, 2,6,13);
        add_edge(S_demo, 3,4,2);
        add_edge(S_demo, 3,5,5);
        add_edge(S_demo, 4,5,1);
        add_edge(S_demo, 4,6,6);
        add_edge(S_demo, 4,7,3);
        add_edge(S_demo, 5,7,1);
        add_edge(S_demo, 6,7,2);
        add_edge(S_demo, 6,8,3);
        add_edge(S_demo, 7,8,6);

        res = shortest_path(S_demo, 1);
        println("Minimum distances:",res["dist"])
        println("Prev:",res["prev"])
        # check the video https://www.youtube.com/watch?v=5GT5hYzjNoo
    end
end