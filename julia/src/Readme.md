# Instructions

The command to execute a single run with the parameters contained in the file _experiments\_config.csv_ is

```julia
julia run_experiments.jl -c experiments_config.csv
```

The command to execute the parameter tuning procedure taking the values for the parameters indicated in the file _experiments\_config.csv_ is

```
julia dsnn_param_tuning.jl -c experiments_config.csv
```



