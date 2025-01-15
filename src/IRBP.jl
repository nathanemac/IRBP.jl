module IRBP

using LinearAlgebra
using ProxTV

# import functions that we extend from ShiftedProximalOperators and ProxTV
import ShiftedProximalOperators.shift!
import ShiftedProximalOperators.shifted
import ShiftedProximalOperators.prox!

import ProxTV.shift!
import ProxTV.shifted
import ProxTV.prox!

export ProjLpBall,
    ShiftedProjLpBall,
    irbp_alg,
    pnorm,
    get_lp_ball_projection,
    get_weightedl1_ball_projection,
    prox!,
    shifted,
    shift!

include("irbp_utils.jl")
include("irbp_alg.jl")

end # module
