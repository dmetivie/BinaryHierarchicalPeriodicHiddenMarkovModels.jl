function update_B!(B::AbstractArray{T,4} where {T}, Ξ³::AbstractMatrix, π::AbstractMatrix{<:Bool}, estimator, idx_tj::Matrix{Vector{Vector{Int}}})
    @argcheck size(Ξ³, 1) == size(π, 1)
    @argcheck size(Ξ³, 2) == size(B, 1)
    N = size(Ξ³, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_memory = size(B, 4)
    ## For periodicHMM only the n π corresponding to B(t) are used to update B(t)

    @inbounds for t in OneTo(T)
        for i in OneTo(K)
            for j = 1:D
                for m = 1:size_memory
                    if sum(Ξ³[idx_tj[t, j][m], i]) > 0
                        B[i, t, j, m] = estimator(Bernoulli, π[idx_tj[t, j][m], j], Ξ³[idx_tj[t, j][m], i])
                    else
                        B[i, t, j, m] = Bernoulli(eps())
                    end
                end
            end
        end
    end
end

function fit_mle!(
    hmm::HierarchicalPeriodicHMM,
    π::AbstractArray{<:Bool},
    π_past::AbstractArray{<:Bool}
    ;
    n2t=n_to_t(size(π, 1), size(hmm, 3))::AbstractVector{<:Integer},
    display=:none,
    maxiter=100,
    tol=1e-3,
    robust=false,
    estimator=fit_mle)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T, size_memory = size(π, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4)
    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])

    # Allocate memory for in-place updates
    c = zeros(N)
    Ξ± = zeros(N, K)
    Ξ² = zeros(N, K)
    Ξ³ = zeros(N, K)
    ΞΎ = zeros(N, K, K)
    LL = zeros(N, K)

    # assign category for observation depending in the π_past π
    lag_cat = conditional_to(π, π_past)
    idx_tj = idx_observation_of_past_cat(lag_cat, n2t, T, size_memory)

    loglikelihoods!(LL, hmm, π, lag_cat; n2t=n2t)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(Ξ±, c, hmm.a, hmm.A, LL; n2t=n2t)
    backwardlog!(Ξ², c, hmm.a, hmm.A, LL; n2t=n2t)
    posteriors!(Ξ³, Ξ±, Ξ²)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, Ξ±, Ξ²)
        update_A!(hmm.A, ΞΎ, Ξ±, Ξ², LL; n2t=n2t)
        update_B!(hmm.B, Ξ³, π, estimator, idx_tj)
    
        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely π.
        robust && (hmm.A .+= eps())
    
        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)
    
        loglikelihoods!(LL, hmm, π, lag_cat; n2t=n2t)
    
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    
        forwardlog!(Ξ±, c, hmm.a, hmm.A, LL; n2t=n2t)
        backwardlog!(Ξ², c, hmm.a, hmm.A, LL; n2t=n2t)
        posteriors!(Ξ³, Ξ±, Ξ²)
    
        logtotp = sum(c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")
    
        push!(history.logtots, logtotp)
        history.iterations += 1
    
        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history.converged = true
            break
        end
    
        logtot = logtotp
    end

    if !history.converged
        if display in [:iter, :final]
            println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
        end
    end

    history
end