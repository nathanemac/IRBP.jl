module IRBP

using LinearAlgebra
using Plots
using ProxTV

# import functions that we extend from ShiftedProximalOperators and ProxTV
import ShiftedProximalOperators.shift!
import ShiftedProximalOperators.shifted
import ShiftedProximalOperators.prox!

import ProxTV.shift!
import ProxTV.shifted
import ProxTV.prox!
import ProxTV.update_prox_context!

export ProjLpBall,
    ShiftedProjLpBall,
    irbp_alg,
    pnorm,
    get_lp_ball_projection,
    get_weightedl1_ball_projection,
    prox!,
    shifted,
    shift!,
    IRBPContext,
    update_prox_context!,
    ModelFunction

include("irbp_utils.jl")
include("irbp_alg.jl")

end # module
