module BinaryHierarchicalPeriodicHiddenMarkovModels

using Distributions
using HMMBase
using HMMBase: posteriors!, vec_maximum, EMHistory, update_a!, isprobvec # function not exported by default by HHMBase
using PeriodicHiddenMarkovModels
using PeriodicHiddenMarkovModels: viterbi
using JuMP, Ipopt
using LsqFit

using ShiftedArrays: lag, lead

using Base: OneTo
using ArgCheck
using Random: AbstractRNG, GLOBAL_RNG
using LogExpFunctions: logsumexp

import Base: ==, copy, size
import HMMBase: fit_mle!, fit_mle
import PeriodicHiddenMarkovModels: rand, forward, backward, forwardlog!, backwardlog!, viterbi, viterbi!, viterbilog!

export
    # periodichmm.jl
    HierarchicalPeriodicHMM,
    sort_wrt_ref!,
    randhierarchicalPeriodicHMM,
    rand,
    # messages.jl
    forward,
    backward,
    # likelihoods.jl
    loglikelihoods,
    likelihoods,
    viterbi,
    # trigonometric
    fit_θ!,
    fit_θ,
    fit_θᴬ!,
    fit_θᴮ!,
    polynomial_trigo,
    Trig2HierarchicalPeriodicHMM,
    # fit slice
    fit_mle_all_slices

include("utilities.jl")

for fname in ["periodichmm.jl", "mle.jl", "likelihoods.jl", "viterbi.jl"], foldername in ["hierarchical_periodic"]
    include(joinpath(foldername, fname))
end

include("trigonometric/mle.jl")
include("trigonometric/trig_utilities.jl")
include("fit_slice/mle.jl")

end