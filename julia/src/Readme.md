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



