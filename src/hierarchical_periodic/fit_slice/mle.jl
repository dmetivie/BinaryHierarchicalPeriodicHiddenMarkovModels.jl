function fit_mle_all_slices(hmm::PeriodicHMM, 𝐘;
    n2t=n_to_t(size(𝐘, 1), size(hmm, 3))::AbstractVector{<:Integer},
    robust=false,
    smooth=true, window=-15:15, kernel=:step,
    history=false,
    kwargs...)

    hmm = copy(hmm)
    N = size(𝐘, 1)
    @argcheck size(𝐘, 1) == size(n2t, 1)

    K, T = size(hmm, 1), size(hmm, 3)
    α = hcat([vec(sum(hmm.A[:, :, t], dims=1) / K) for t = 1:T]...)
    n_in_t = [findall(n2t .== t) for t = 1:T] #
    hist = Vector{HMMBase.EMHistory}(undef, T)
    cycle = CyclicArray(1:T, "1D")
    for t = 1:T
        n_in_t_extanded = sort(vcat([n_in_t[tt] for tt in cycle[[t - 12, t - 6, t, t + 6, t + 12]]]...)) # extend dataset
        hist[t] = fit_mle_B_slice!(@view(α[:, t]), @view(hmm.B[:, t]), 𝐘[n_in_t_extanded, :]; kwargs...)
        # hist[t] = fit_mle_B_slice!(@view(α[:,t]), @view(hmm.B[:,t]), 𝐘[n_in_t[t],:]; kwargs...)
    end

    LL = zeros(N, K)

    if smooth == true
        smooth_B = lagou(hmm.B, dims=2, window=window, kernel=kernel)
        smooth_α = lagou(α, dims=2, window=window, kernel=kernel)

        # evaluate likelihood for each type k
        loglikelihoods!(LL, smooth_B, 𝐘, n2t)
        [LL[n, k] += log(smooth_α[k, n2t[n]]) for k = 1:K, n = 1:N]
    else
        # evaluate likelihood for each type k
        loglikelihoods!(LL, hmm.B, 𝐘, n2t)
        [LL[n, k] += log(α[k, n2t[n]]) for k = 1:K, n = 1:N]
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust=robust)

    return history ? (hmm, hist) : hmm
end

function fit_mle_all_slices(hmm::HierarchicalPeriodicHMM, 𝐘::AbstractArray{<:Bool}, 𝐘_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(𝐘, 1), size(hmm, 3))::AbstractVector{<:Integer},
    robust=false,
    smooth=true, window=-15:15, kernel=:step,
    history=false,
    kwargs...)

    hmm = copy(hmm)

    N, K, T = size(𝐘, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(𝐘, 1) == size(n2t, 1)

    # assign category for observation depending in the past 𝐘
    lag_cat = conditional_to(𝐘, 𝐘_past)

    α = hcat([vec(sum(hmm.A[:, :, t], dims=1) / K) for t = 1:T]...)
    n_in_t = [findall(n2t .== t) for t = 1:T] #
    hist = Vector{EMHistory}(undef, T)
    # cycle(t) = t % 366
    for t = 1:T
        n_in_t_extanded = n_in_t#sort(vcat([n_in_t[tt] for tt in cycle[[t - 12, t - 7, t, t + 6, t + 13]]]...)) # extend dataset
        hist[t] = fit_mle_B_slice!(@view(α[:, t]), @view(hmm.B[:, t, :, :]), 𝐘[n_in_t_extanded, :], lag_cat[n_in_t_extanded, :]; kwargs...)
    end

    LL = zeros(N, K)

    if smooth == true
        smooth_B = lagou(hmm.B, dims=2, window=window, kernel=kernel)
        smooth_α = lagou(α, dims=2, window=window, kernel=kernel)
        loglikelihoods!(LL, smooth_B, 𝐘, n2t, lag_cat)
        for k = 1:K, n = 1:N
            LL[n, k] += log(smooth_α[k, n2t[n]])
        end
    else
        loglikelihoods!(LL, hmm.B, 𝐘, n2t, lag_cat)
        for k = 1:K, n = 1:N
            LL[n, k] += log(α[k, n2t[n]])
        end
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust=robust)

    return history ? (hmm, hist) : hmm
end


function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,2} where {F<:MultivariateDistribution}, 𝐘, n2t::AbstractArray{Int})
    N, K = size(𝐘, 1), size(B, 1)
    @argcheck size(LL) == (N, K)

    for n in OneTo(N)
        t = n2t[n]
        for k in OneTo(K)
            LL[n, k] = logpdf(B[k, t], 𝐘[n, :])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,2} where {F<:UnivariateDistribution}, 𝐘, n2t::AbstractArray{Int})
    N, K = size(𝐘, 1), size(B, 1)
    @argcheck size(LL) == (N, K)

    for n in OneTo(N)
        t = n2t[n]
        for k in OneTo(K)
            LL[n, k] = logpdf(B[k, t], 𝐘[n])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,4} where {F<:UnivariateDistribution}, 𝐘, n2t::AbstractArray{Int}, lag_cat::Matrix{Int})
    N, K, D = size(𝐘, 1), size(B, 1), size(𝐘, 2)
    @argcheck size(LL) == (N, K)

    for n in OneTo(N)
        t = n2t[n]
        for k in OneTo(K)
            LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, t, 1:D, lag_cat[n, :])]), 𝐘[n, :])
        end
    end
end

function fit_mle_A_from_slice!(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL, n2t; robust=false)

    N, K = size(LL)
    T = size(A, 3)
    c = zeros(N)
    γ = zeros(N, K)

    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)
    a[:] = γ[1, :] # initial date
    max_aposteriori = argmaxrow(γ)
    max_aposteriori_next = lead(max_aposteriori)

    Q = zeros(K, K, T)
    for n = 1:N
        t = n2t[n]
        if max_aposteriori_next[n] === missing
            continue
        else
            Q[max_aposteriori[n], max_aposteriori_next[n], t] += 1
        end
    end
    robust && (Q .+= eps())
    [A[:, :, t] = Q[:, :, t] ./ sum(Q[:, :, t], dims=2) for t = 1:T]
end

function fit_mle_B_slice!(α::AbstractVector, B::AbstractVector{F} where {F<:Distribution}, 𝐘;
    rand_ini=true,
    n_random_ini=10, display_random=false,
    Dirichlet_α=0.8, Dirichlet_categories=0.85,
    ref_station=1, kwargs...)
    if rand_ini == true
        α[:], B[:], h = fit_em_multiD_rand(α, B, 𝐘; n_random_ini=n_random_ini, Dirichlet_α=Dirichlet_α, Dirichlet_categories=Dirichlet_categories, display_random=display_random, kwargs...)
    else
        h = fit_em_multiD!(α, B, 𝐘; kwargs...)
    end
    sort_wrt_ref!(α, B, ref_station)
    h
end

function fit_mle_B_slice!(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    𝐘::Matrix{Int64}, lag_cat::Matrix{Int};
    rand_ini=true,
    n_random_ini=10, display_random=false,
    Dirichlet_α=0.8, Dirichlet_categories=0.85,
    ref_station=1, kwargs...)

    K, size_memory = size(B, 1), size(B, 3)
    idx_j = idx_observation_of_past_cat(lag_cat, K, size_memory)

    if rand_ini == true
        α[:], B[:], h = fit_em_multiD_rand(α, B, 𝐘, lag_cat, idx_j; n_random_ini=n_random_ini, Dirichlet_α=Dirichlet_α, display_random=display_random, kwargs...)
    else
        h = fit_em_multiD!(α, B, 𝐘, lag_cat, idx_j; kwargs...)
    end
    sort_wrt_ref!(α, B, ref_station)
    h
end
