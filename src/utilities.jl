argmaxrow(A::AbstractMatrix{<:Real}) = [argmax(A[i, :]) for i = 1:size(A, 1)]

function cycle(t, T)
    if 1 ≤ t ≤ T
        return t
    elseif T < t ≤ 2T
        return t - T
    elseif -T < t ≤ 0
        return t + T
    else
        println("Not implemented")
    end
end

n_per_category(s, h, t, y, n_in_t, n_occurence_history) = (n_in_t[t] ∩ n_occurence_history[s, h, y])

bin2digit(x) = sum(x[length(x)-i+1] * 2^(i - 1) for i = 1:length(x)) + 1
bin2digit(x::Tuple) = bin2digit([x...])

function dayx(lag_obs::AbstractArray)
    memory = length(lag_obs)
    t = tuple.([lag_obs[m] for m = 1:memory]...)
    bin2digit.(t)
end

function conditional_to(𝐘::AbstractArray{<:Bool}, 𝐘_past::AbstractArray{<:Bool})
    memory = size(𝐘_past, 1)
    if memory == 0
        return ones(Int, size(𝐘))
    else
        lag_obs = [copy(lag(𝐘, m)) for m = 1:memory]  # transform dry=0 into 1 and wet=1 into 2 for array indexing
        for m = 1:memory
            lag_obs[m][1:m, :] .= reverse(𝐘_past[1:m, :], dims=1) # avoid the missing first row
        end
        return dayx(lag_obs)
    end
end

function idx_observation_of_past_cat(lag_cat, n2t, T, size_memory)
    # Matrix(T,D) of vector that give the index of data of same 𝐘_past.
    # ie. size_memory = 1 (no memory) -> every data is in category 1
    # ie size_memory = 2 (memory on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_tj = Matrix{Vector{Vector{Int}}}(undef, T, D)
    n_in_t = [findall(n2t .== t) for t = 1:T] # could probably be speeded up e.g. recusivly suppressing already classified label with like sortperm
    for t in OneTo(T)
        n_t = n_in_t[t]
        for j = 1:D
            n_tm = [findall(lag_cat[n_t, j] .== m) for m = 1:size_memory]
            idx_tj[t, j] = [n_t[n_tm[m]] for m = 1:size_memory]
            ##
        end
    end
    return idx_tj
end

function idx_observation_of_past_cat(lag_cat, size_memory)
    # Matrix(T,D) of vector that give the index of data of same past.
    # ie. size_memory = 1 (no memory) -> every data is in category 1
    # ie size_memory = 2 (memory on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_j = Vector{Vector{Vector{Int}}}(undef, D)
    for j = 1:D
        idx_j[j] = [findall(lag_cat[:, j] .== m) for m = 1:size_memory]
    end
    return idx_j
end

random_product_Bernoulli(D::Int, K::Int, size_memory::Int) = [Bernoulli(rand()) for k = 1:K, j = 1:D, m = 1:size_memory]

#TODO add to rand(HierarchicalPeriodicHMM, blabla)
function randhierarchicalPeriodicHMM(K, T, D, size_memory; ref_station=1, ξ=ones(K) / K)
    B_rand = Bernoulli.(rand(K, T, D, size_memory))  # completly random -> bad
    Q_rand = zeros(K, K, T)
    for t in 1:T
        Q_rand[:, :, t] = randtransmat(K) # completly random -> bad
    end
    hmm_random = HierarchicalPeriodicHMM(ξ, Q_rand, B_rand)
    sort_wrt_ref!(hmm_random, ref_station)
    return hmm_random
end
# TODO: site dependent memory #
# function conditional_to(𝐘::AbstractArray, memory::AbstractVector;
#     𝐘_past=[0 1 0 1 1 0 1 0 0 0
#         1 1 0 1 1 1 1 1 1 1
#         1 1 0 1 1 1 0 1 1 1
#         1 1 0 1 1 0 0 0 1 0
#         1 1 0 1 1 0 0 1 0 1]
# )
#     D = size(𝐘, 2)
#     lag_cat = zeros(Int, size(𝐘))
#     for j = 1:D
#         lag_cat[:, j] = conditional_to(𝐘[:, j], memory[j], 𝐘_past=𝐘_past[:, j])
#     end
#     return lag_cat
# end