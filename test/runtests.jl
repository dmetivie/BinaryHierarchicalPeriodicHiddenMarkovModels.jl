using BinaryHierarchicalPeriodicHiddenMarkovModels
using Test

@testset "BinaryHierarchicalPeriodicHiddenMarkovModels.jl" begin
    K = 4
    T = 20
    D = 10
    N = 100
    ξ = [1; zeros(K-1)]
    ref_station = 1

    # Test that the HMM is well definied with different order of chain (= "local memory" in my jargon)
    for order in 0:3
        hmm_random = randhierarchicalPeriodicHMM(K, T, D, order; ξ=ξ, ref_station=ref_station)

        z, y = rand(hmm_random, N, seq = true, z_ini = 1, y_ini = zeros(Int, order, D))

        y = rand(hmm_random, N)
    end

    #TODO add test comparing order = 0 to PeriodicHMM (it should be exactly the same)
end