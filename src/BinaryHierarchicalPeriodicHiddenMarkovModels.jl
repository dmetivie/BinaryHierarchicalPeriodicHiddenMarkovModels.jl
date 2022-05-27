module BinaryHierarchicalPeriodicHiddenMarkovModels

using Distributions
using HMMBase
using HMMBase: posteriors!, vec_maximum, EMHistory, update_a!, isprobvec # function not exported by default by HHMBase
using PeriodicHiddenMarkovModels
using JuMP, Ipopt
using LsqFit

using ShiftedArrays: lag, lead

using Base: OneTo
using ArgCheck
using Random: AbstractRNG, GLOBAL_RNG

import Base: ==, copy, rand, size
import HMMBase: fit_mle!

export
    # periodichmm.jl
    HierarchicalPeriodicHMM,
    sort_wrt_ref!,
    # messages.jl
    forward,
    backward,
    posteriors,
    # likelihoods.jl
    loglikelihoods,
    likelihoods,
    # mle.jl
    fit_mle,
    # trigonometric
    fit_Q!,
    fit_Y!,
    polynomial_trigo

include("utilities.jl")

for fname in ["periodichmm.jl", "mle.jl", "likelihoods.jl"], foldername in ["hierarchical_periodic"]
    include(joinpath(foldername, fname))
end

include("trigonometric/mle.jl")


end