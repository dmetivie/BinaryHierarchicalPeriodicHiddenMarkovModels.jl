#! TODO fix convention 𝐘 size(𝐘) = D, N not the opposite. (Here it does not change)
function likelihoods!(L::AbstractMatrix, hmm::HierarchicalPeriodicHMM, 𝐘::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(L, 1), size(hmm, 3))::AbstractVector{<:Integer})
    N, K, D = size(𝐘, 1), size(hmm, 1), size(hmm, 2)
    @argcheck size(L) == (N, K)

    for i in OneTo(K)
        for n in OneTo(N)
            t = n2t[n] # periodic t
            L[n, i] = pdf(product_distribution(hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])]), 𝐘[n, :])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::HierarchicalPeriodicHMM, 𝐘::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer})
    N, K, D = size(𝐘, 1), size(hmm, 1), size(hmm, 2)
    @argcheck size(LL) == (N, K)

    for i in OneTo(K)
        for n in OneTo(N)
            t = n2t[n]
            LL[n, i] = logpdf(product_distribution(hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])]), 𝐘[n, :])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,2} where {F<:MultivariateDistribution}, 𝐘::AbstractMatrix; n2t=n_to_t(size(𝐘, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K = size(𝐘, 1), size(B, 1)
    @argcheck size(LL) == (N, K)

    for n in OneTo(N)
        t = n2t[n]
        for k in OneTo(K)
            LL[n, k] = logpdf(B[k, t], 𝐘[n, :])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,2} where {F<:UnivariateDistribution}, 𝐘::AbstractVector; n2t=n_to_t(size(𝐘, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K = size(𝐘, 1), size(B, 1)
    @argcheck size(LL) == (N, K)

    for k in OneTo(K)
        for n in OneTo(N)
            t = n2t[n]
            LL[n, k] = logpdf(B[k, t], 𝐘[n])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,4} where {F<:UnivariateDistribution}, 𝐘::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(𝐘, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K, D = size(𝐘, 1), size(B, 1), size(𝐘, 2)
    @argcheck size(LL) == (N, K)

    for k in OneTo(K)
        for n in OneTo(N)
            t = n2t[n]
            LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, t, 1:D, lag_cat[n, :])]), 𝐘[n, :])
        end
    end
end
# * Bayesian Criterion * #

#!TODO: change `lag_cat = conditional_to(y, memory)` to `lag_cat = conditional_to(y, y_past)`
function complete_loglikelihood(hmm::HierarchicalPeriodicHMM, y::AbstractArray, z::AbstractVector, n2t::AbstractArray)
    N, size_memory, D = size(y, 1), size(hmm, 4), size(y, 2)
    memory = Int(log(size_memory) / log(2))
    lag_cat = conditional_to(y, memory)

    return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1) + sum(logpdf(product_distribution(hmm.B[CartesianIndex.(z[n], n2t[n], 1:D, lag_cat[n, :])]), y[n, :]) for n = 1:N)
end

function complete_loglikelihood(hmm::PeriodicHMM, y::AbstractArray, z::AbstractVector, n2t::AbstractArray)
    N, D = size(y, 1), size(y, 2)

    return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1) + sum(logpdf(product_distribution(hmm.B[CartesianIndex.(z[n], n2t[n], 1:D)]), y[n, :]) for n = 1:N)
end

function complete_loglikelihood(hmm::HMM, y::AbstractArray, z::AbstractVector, n2t::AbstractArray)
    N, D = size(y, 1), size(y, 2)

    return sum(log(hmm.A[z[n], z[n+1]]) for n = 1:N-1) + sum(logpdf(product_distribution(hmm.B[CartesianIndex.(z[n], 1:D)]), y[n, :]) for n = 1:N)
end
