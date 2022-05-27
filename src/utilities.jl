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


function idx_observation_of_past_cat(lag_cat, n2t, T, K, size_memory)
    # Matrix(T,D) of vector that give the index of data of same 𝐘_past.
    # ie. size_memory = 1 (no memory) -> every data is in category 1
    # ie size_memory = 2 (memory on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_tj = Matrix{Vector{Vector{Int}}}(undef, T, D)
    n_in_t = [findall(n2t .== t) for t = 1:T] # could probably be speeded up e.g. recusivly suppressing already classified label with like sortperm
    @inbounds for t in OneTo(T)
        n_t = n_in_t[t]
        for i in OneTo(K)
            for j = 1:D
                # apparently the two following takes quite long ~29ms for all the loops
                # TODO improve time!
                n_tm = [findall(lag_cat[n_t, j] .== m) for m = 1:size_memory]
                idx_tj[t, j] = [n_t[n_tm[m]] for m = 1:size_memory]
                ##
            end
        end
    end
    return idx_tj
end


#* Trigo part *#

function polynomial_trigo(t::Number, β; T=366)
    d = (length(β) - 1) ÷ 2
    if d == 0
        return β[1]
    else
        f = 2π / T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] + sum(β[2*l] * cos(f * l * t) + β[2*l+1] * sin(f * l * t) for l = 1:d)
    end
end

function polynomial_trigo(t::AbstractArray, β; T=366)
    d = (length(β) - 1) ÷ 2
    if d == 0
        return β[1]
    else
        f = 2π / T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] .+ sum(β[2*l] * cos.(f * l * t) + β[2*l+1] * sin.(f * l * t) for l = 1:d)
    end
end

interleave2(args...) = collect(Iterators.flatten(zip(args...))) # merge two vector with alternate elements

function fit_Q!(p::AbstractArray, A::AbstractArray{N,2} where {N}; silence=true)
    T, K = size(A, 2), size(A, 1)
    @assert K - 1 == size(p, 1)
    d = (size(p, 2) - 1) ÷ 2
    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]

    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    @variable(model, p_jump[k=1:(K-1), j=1:(2d+1)])
    set_start_value.(p_jump, p)
    # Polynomial P_kl

    @NLexpression(model, Pol[t=1:T, k=1:K-1], sum(trig[t][j] * p_jump[k, j] for j = 1:length(trig[t])))

    @NLobjective(
        model,
        Min,
        sum((A[k, t] - exp(Pol[t, k]) / (1 + sum(exp(Pol[t, l]) for l = 1:K-1)))^2 for k = 1:K-1, t = 1:T)
        +
        sum((A[K, t] - 1 / (1 + sum(exp(Pol[t, l]) for l = 1:K-1)))^2 for t = 1:T)
    )
    optimize!(model)
    p[:, :] = value.(p_jump)
end

m_Bernoulli(t, p; T=366) = 1 ./ (1 .+ exp.(polynomial_trigo(t, p; T=T)))
# Fit (faster than JuMP) with LsqFit
function fit_Y!(p::AbstractVector, B::AbstractVector)
    T = size(B, 1)
    p[:] = curve_fit((t, p) -> m_Bernoulli(t, p, T=T), collect(1:T), B, convert(Vector, p)).param
end