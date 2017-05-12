nofworkers = 4
addprocs(nofworkers)
results = cell(nofworkers)

# worker code
@everywhere function talk2me(N,start)
    cnt = start
    for i = collect(1:N)
        cnt += i
    end
    return cnt
end


@time begin
    @sync for (idx, pid) in enumerate(workers())
        println(idx, pid)
        @async results[idx] = remotecall_fetch(pid, talk2me, 10000, pid) 
    end
end
println(results)
