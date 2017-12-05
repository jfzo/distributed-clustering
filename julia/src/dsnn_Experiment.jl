module DSNN_EXPERIMENT

using DSNN_IO
using DSNN_SNN
using Graphs
using LightGraphs
using Clustering

"""
    config_as_str(config)

Returns a ';' separated string containing all pairs key:value of the configuration dict.
"""
function config_as_str(config::Dict{String, Any})
    s = "";
    fst_elem = true;
    for (k, val) in map(x->(x, config[x]), sort(collect(keys(config) ) ) )
        if fst_elem
            s = string(k,":",val);
            fst_elem = false;
        else
            s = string(s,";",k,":",val);
        end
    end
    return s;
end

function perform_corepoint_snn(adj_mat::SparseMatrixCSC{Float64,Int64}, config::Dict{String, Any})
    println("Applying SNN-Clustering over the corepoints...");
    cp_results = DSNN_SNN.snn_clustering(config["master.snn.eps"], config["master.snn.minpts"], adj_mat);

    labels_found = fill(0, size(cp_results["labels"],1));
    for c in collect(1:size(cp_results["labels"],2))
        for i in cp_results["labels"][:,c].nzind
            labels_found[i] = cp_results["clusters"][c]; # extract the right assigned label name
        end
    end

    println("Num. Clusters found:",length(unique(labels_found)))
    if length(find(x->x<0, cp_results["clusters"])) > 0
        println(@sprintf("Percentage of noise:: %0.2f", 
                length(cp_results["labels"][:,1].nzind)/size(cp_results["labels"],1)))
    else
        println("Amount of noise: 0");
    end
    writedlm(@sprintf("%s.corepoints.snn.labels",config["master.inputpath"]), labels_found, "\n");
    println("Labels stored in file ", @sprintf("%s.corepoints.snn.labels",config["master.inputpath"]) );
end


function perform_corepoint_conncomps(adj_mat::SparseMatrixCSC{Float64,Int64}, config::Dict{String, Any})
    println("Applying Connected Components over the corepoints...");
    numpoints = size(adj_mat,2);

    G = Graphs.simple_adjlist(numpoints, is_directed=false);
    for i in collect(1:numpoints)
        for j in adj_mat[:,i].nzind
            Graphs.add_edge!(G, i, j)
        end
    end

    cmps = Graphs.connected_components(G);

    println("Num. connected components:",length(cmps));
    labels_found = fill(-1, numpoints);
    for cmp_i in eachindex(cmps)
        for p in cmps[cmp_i]
            labels_found[p] = cmp_i;
        end
    end
    println("Num. Clusters found:",length(unique(labels_found)))
    writedlm(@sprintf("%s.corepoints.conncomps.labels", config["master.inputpath"]), labels_found, "\n");
    println("Labels stored in file ", @sprintf("%s.corepoints.conncomps.labels",config["master.inputpath"]) );
end


function perform_corepoint_maxcliques(adj_mat::SparseMatrixCSC{Float64,Int64}, config::Dict{String, Any})
    println("Applying Maximal Clique over the corepoints...");
    numpoints = size(adj_mat,2);
    println("Num. points:",numpoints);

    G = Graphs.simple_adjlist(numpoints, is_directed=false);
    for i in collect(1:numpoints)
        for j in adj_mat[:,i].nzind
            Graphs.add_edge!(G, i, j)
        end
    end

    cmps = Graphs.maximal_cliques(G);

    println("Num. Cliques:",length(cmps));
    labels_found = fill(-1, numpoints);
    for cmp_i in eachindex(cmps)
        for p in cmps[cmp_i]
            labels_found[p] = cmp_i;
        end
    end

    println("Num. Clusters found:",length(unique(labels_found)))
    writedlm(@sprintf("%s.corepoints.cliques.labels", config["master.inputpath"]), labels_found, "\n");
    println("Labels stored in file ", @sprintf("%s.corepoints.cliques.labels",config["master.inputpath"]) );
end



function perform_corepoint_lblprop(adj_mat::SparseMatrixCSC{Float64,Int64}, config::Dict{String, Any})
    println("Applying Label propagation over the corepoints...")

    numpoints = size(adj_mat,2);
    G = LightGraphs.Graph(numpoints)
    for i in collect(1:numpoints)
       for j in adj_mat[:, i].nzind
           if j > i 
               # maybe a threshold based on adj_mat[j,i] could be used !
               if ~LightGraphs.add_edge!(G, i, j)
                   println("[M] Error: Cannot add edge between vertices ",i," and ",j)
               end
           end
       end
    end
    labels_found, conv_history = LightGraphs.label_propagation(G);
    println("Num. Clusters found:",length(unique(labels_found)))
    writedlm(@sprintf("%s.corepoints.lblprop.labels",config["master.inputpath"]), labels_found, "\n");
    println("Labels stored in file ", @sprintf("%s.corepoints.lblprop.labels",config["master.inputpath"]) );
end



function perform_corepoint_dbscan(adj_mat::SparseMatrixCSC{Float64,Int64}, config::Dict{String, Any})
    println("Applying DBSCAN over the corepoints...");

    #dbscan_cl = Clustering.dbscan(full(Dw), 0.1, min_neighbors=15);
    dbscan_cl = Clustering.dbscan(full(1.0 .- adj_mat), config["master.dbscan.eps"], config["master.dbscan.minpts"]);
    labels_found = dbscan_cl.assignments;
    println("Num. Clusters found:",length(unique(labels_found)))
    writedlm(@sprintf("%s.corepoints.dbscan.labels", config["master.inputpath"]), labels_found, "\n");
    println("Labels stored in file ",@sprintf("%s.corepoints.dbscan.labels", config["master.inputpath"]));
end


end
