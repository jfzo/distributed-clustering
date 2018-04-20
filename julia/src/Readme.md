# Instructions

The command to execute a single run with the parameters contained in the file _experiments\_config.csv_ is

```julia
julia run_experiments.jl -c experiments_config.csv
```

The command to execute the parameter tuning procedure taking the values for the parameters indicated in the file _experiments\_config.csv_ is

```
julia dsnn_param_tuning.jl -c experiments_config.csv
```

Some useful commands:

```
To sort the ARI scores (ascending order):

sed -n '/D-SNN/s/ \+/ /gp' tuning_progress.log |cut -d ' ' -f5|sort -g


To display the experiments in which the score XXX was attained (for instance the last score shown by the command above):

grep -B 8 -A 5 XXX tuning_progress.log
```


# Available Parameters and their definitions

| Parameter | Sample value | Definition |
|-----------|--------------|------------|
| logging.path:str | ./tuning_progress.log | File in which the output information is dump |
| l2knng.path:str | /workspace/l2knng/build/knng | Path to the L2KNNG program |
| benchmark:str | /workspace/RCV1/rcv1_meta.dat.clustering.4 | File containing labels estimated by the benchmark method |
| master.inputpath:str | /workspace/RCV1/rcv1_meta.dat | Path where the input data vectors are stored |
| master.dbscan.eps:float | 0.5 | Used only when stage2clustering is set to dbscan |
| master.dbscan.minpts:int | 3 | Used only when stage2clustering is set to dbscan |
| master.nodelist:str_list | 158.251.93.5:3301,158.251.93.5:3302,158.251.93.5:3303 | Comma separated list of node IP 's |
| master.snn.eps:float | 0.001 | Used only when stage2clustering is set to snn |
| master.snn.minpts:int | 7 | Used only when stage2clustering is set to snn |
| master.stage2clustering:str | conncomps | Strategy employed to label the corepoints reported by the workers at the Master |
| master.stage2knn:int | 3 | Size of the neighborhood used to build the SNN similarity matrix over the retrieved points from  all workers |
| master.stage2snnsim_threshold:float | 0.0 |  |
| master.use_snngraph:bool | false | Generate (at the Master) a snn graph connecting objects that are contained in eachother neighborhood or to use only the snn similarity matrix. |
| worker.knn:int | 120 | Employed in stage1 and stage2 at the workers for building the SNN similarity matrix |
| worker.sample_pct:float | 0.050000 | Employed as the size of the sample that each worker generates at the end of stage1 |
| worker.snn_eps:float | 1e-4 | Employed in stage1 and stage2 at the workers for performing the SNN clustering and for detecting noisy points in stage2 |
| worker.snn_minpts:int | 2 | Employed in stage1 at each worker to perform the SNN clustering |
| worker.use_snngraph:bool | true | Generate (at each Worker) a snn graph connecting objects that are contained in eachother neighborhood or to use only the snn similarity matrix |
| worker.use_snnclustering:bool | false | Indicates if SNN clustering is employed to find the corepoints and the associated sampled points at the Stage-1. If set to __False__, epsilon-nn is employed instead |
| worker.coredetection_eps:float | 2e-1 | Used when worker.use_snnclustering is set to FALSE (epsilon-nn corepoint detection)|
| worker.coredetection_minpts:int | 100 | Used when worker.use_snnclustering is set to FALSE (epsilon-nn corepoint detection)|
| seed:int | 20929911109928830 | Seed employed in all randomized operations |
