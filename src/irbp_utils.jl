# This file contains utility functions used in the package.


##################################
# 1. Functions used in pure IRBP #
##################################
"""
    generate_data(n::Int; μ=0.0, σ=1.0) = randn(n) * σ .+ μ
    Generates n random numbers from a normal distribution with mean μ and standard deviation σ.
"""
function generate_data(n::Int; μ = 0.0, σ = 1.0)
    return randn(n) * σ .+ μ
end


"""
    least_square_loss(x, y)
    Compute the least square loss between x : the current solution and y : the point to be projected.
"""
function least_square_loss(x, y)
    return norm(x - y, 2)^2
end


"""
    pnorm(x, p)

Custom p-norm for real p:
  pnorm(x, p) = (sum(abs.(x).^p))^(1/p)
"""
function pnorm(x::AbstractVector{T}, p::T) where {T<:Real}
    s = 0.0
    @inbounds for i = 1:length(x)
        s += abs(x[i])^p
    end
    return s^(1 / p)
end


"""
    plot_weightedl1_ball_2D(weights, radius; color=:blue)
    Trace la boule L1 pondérée définie par
       w1*|x1| + w2*|x2| <= radius
    (forme de losange en 2D).
"""
function plot_weightedl1_ball_2D(weights, radius; color = :blue)
    w1, w2 = weights

    # Les 4 sommets du losange
    c1 = (radius / w1, 0.0)
    c2 = (0.0, radius / w2)
    c3 = (-radius / w1, 0.0)
    c4 = (0.0, -radius / w2)

    # On ferme le polygone en bouclant sur le premier sommet
    x_vals = [c1[1], c2[1], c3[1], c4[1], c1[1]]
    y_vals = [c1[2], c2[2], c3[2], c4[2], c1[2]]

    plt = plot(
        x_vals,
        y_vals,
        seriestype = :shape,                 # pour dessiner la forme remplie
        fillalpha = 0.2,
        fillcolor = color,
        linecolor = color,
        legend = false,
        aspect_ratio = :equal,
        title = "Projection sur la boule L1 pondérée",
    )
    return plt
end

function plot_lp_ball_2D(p::Float64, radius::Float64; color = :blue, npoints = 200)
    thetas = range(0, 2π, length = npoints)
    x_coords = similar(thetas)
    y_coords = similar(thetas)

    for (i, t) in enumerate(thetas)
        alpha = radius / (abs(cos(t))^p + abs(sin(t))^p)
        x1 = sign(cos(t)) * (alpha * abs(cos(t))^p)^(1 / p)
        x2 = sign(sin(t)) * (alpha * abs(sin(t))^p)^(1 / p)
        x_coords[i] = x1
        y_coords[i] = x2
    end

    plt = plot(
        x_coords,
        y_coords,
        linecolor = color,
        fillalpha = 0.15,
        fillrange = 0,
        fillcolor = color,
        label = "Lp pseudo-ball (p=$p)",
        aspect_ratio = :equal,
        legend = :topleft,
    )
    return plt
end

#################################################################
# 2. Functions used to communicate with RegularizedOptimization #
#################################################################

mutable struct IRBPContext
    p::Float64
    radius::Float64
    iters_prox_projLp::Int64
    flag_projLp::Int64
    κξ::Float64
    dualGap::Float64
    prox_stats::Vector{Int64}
    shift::Vector{Float64}
    s_k_unshifted::Vector{Float64}
    hk::Float64
    ∇fk::Vector{Float64}
    q_shifted::Vector{Float64}

    # in irbp_alg
    x_ini::Vector{Float64}
    rand_num::Vector{Float64}
    epsilon_ini::Vector{Float64}

    # in get_lp_ball_projection
    signum_vals::Vector{Float64}
    yAbs::Vector{Float64}
    s_k::Vector{Float64}
    temp_vec::Vector{Float64}
    weights::Vector{Float64}
    ν::Float64

    # in get_weightedl1_ball_projection
    signum_vals_l1::Vector{Float64}
    point_to_be_projected_l1::Vector{Float64}
    act_ind_l1::Vector{Bool}
    point_to_be_projected_act_l1::Vector{Float64}
    weights_act_l1::Vector{Float64}
    x_sol_hyper_clamped_l1::Vector{Float64}
    x_opt_l1::Vector{Float64}

    # in get_hyperplane_projection
    s_sub::Vector{Float64} # buffer result
end

# Fonction that creates an IRBPContext object
function IRBPContext(
    n::Int64,
    p::Float64,
    radius::Float64;
    iters_prox_projLp = 100,
    flag_projLp = 0,
    κξ = 0.75,
    dualGap = 1e-8,
)
    shift = zeros(n)
    q_shifted = zeros(n)
    s_k_unshifted = zeros(n)
    hk = 0.0
    ∇fk = zeros(n)
    prox_stats = zeros(Int64, 3)
    x_ini = zeros(n)
    rand_num = zeros(n)
    epsilon_ini = zeros(n)
    signum_vals = zeros(n)
    yAbs = zeros(n)
    s_k = zeros(n)
    temp_vec = zeros(n)
    weights = zeros(n)
    signum_vals_l1 = zeros(n)
    point_to_be_projected_l1 = zeros(n)
    act_ind_l1 = trues(n)
    point_to_be_projected_act_l1 = zeros(n)
    weights_act_l1 = zeros(n)
    x_sol_hyper_clamped_l1 = zeros(n)
    x_opt_l1 = zeros(n)
    s_sub = zeros(n)
    ν = 1.0
    return IRBPContext(
        p,
        radius,
        iters_prox_projLp,
        flag_projLp,
        κξ,
        dualGap,
        prox_stats,
        shift,
        s_k_unshifted,
        hk,
        ∇fk,
        q_shifted,
        x_ini,
        rand_num,
        epsilon_ini,
        signum_vals,
        yAbs,
        s_k,
        temp_vec,
        weights,
        ν,
        signum_vals_l1,
        point_to_be_projected_l1,
        act_ind_l1,
        point_to_be_projected_act_l1,
        weights_act_l1,
        x_sol_hyper_clamped_l1,
        x_opt_l1,
        s_sub,
    )
end


"""
    ProjLpBall(λ, p, radius, context)
    Constructor for the ProjLpBall object.
"""
mutable struct ProjLpBall{R<:Real}
    λ::R         # Regularization parameter, equal to 1 in this case
    p::R         # p-norm with 0 < p < 1
    radius::R    # Radius of the p-ball
    context::IRBPContext # Context object

    function ProjLpBall(λ::R, p::R, radius::R, context::IRBPContext) where {R<:Real}
        @assert p < 1.0 "The p-norm must be < 1.0"
        @assert p > 0.0 "The p-norm must be > 0.0"
        @assert radius > 0.0 "The radius must be > 0."
        @assert λ > 0.0 "The λ parameter must be > 0."
        return new{R}(λ, p, radius, context)
    end
end

"""
    ProjLpBall(λ, p, radius, n)
    Constructor for the ProjLpBall object.
"""
function ProjLpBall(λ, p, radius, n)
    context = IRBPContext(n, p, radius)
    return ProjLpBall(λ, p, radius, context)
end

"""
    (h::ProjLpBall)(x::AbstractVector)
    Indicator function for the p-ball.
    Returns zero if the point is inside the ball, Inf otherwise.
    A small ϵ is added to the radius to avoid numerical issues.
    This function is the "h" function in the RegularizedOptimization.jl framework.
"""
function (h::ProjLpBall)(x::AbstractVector; ϵ::Real = eps()^(1 / 2))
    return indicator_function(x, h.p, h.radius; ϵ = ϵ)
end

function indicator_function(
    x::AbstractVector,
    p::Real,
    radius::Real;
    ϵ::Real = eps()^(1 / 2),
)
    pnorm(x, p)^(p) <= (radius + ϵ) ? 0.0 : Inf
end

mutable struct ShiftedProjLpBall{
    R<:Real,
    V0<:AbstractVector{R},
    V1<:AbstractVector{R},
    V2<:AbstractVector{R},
} <: InexactShiftedProximableFunction
    h::ProjLpBall{R}
    xk::V0
    sj::V1
    sol::V2
    shifted_twice::Bool
    xsy::V2

    function ShiftedProjLpBall(
        h::ProjLpBall{R},
        xk::AbstractVector{R},
        sj::AbstractVector{R},
        shifted_twice::Bool,
    ) where {R<:Real}
        sol = similar(xk)
        xsy = similar(xk)
        new{R,typeof(xk),typeof(sj),typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
    end
end

"""
    shifted(h::ProjLpBall, xk::AbstractVector)

Creates a ShiftedProjLpBall object with initial shift `xk`.
"""
shifted(h::ProjLpBall{R}, xk::AbstractVector{R}) where {R<:Real} =
    ShiftedProjLpBall(h, xk, zero(xk), false)

"""
    shifted(ψ::ShiftedProjLpBall, sj::AbstractVector)

Creates a ShiftedProjLpBall object by adding a second shift `sj`.
"""
shifted(
    ψ::ShiftedProjLpBall{R,V0,V1,V2},
    sj::AbstractVector{R},
) where {R<:Real,V0<:AbstractVector{R},V1<:AbstractVector{R},V2<:AbstractVector{R}} =
    ShiftedProjLpBall(ψ.h, ψ.xk, sj, true)

# Functions to get the name, expression, and parameters of the function
fun_name(ψ::ShiftedProjLpBall) = "shifted Lp norm ball indicator with 0 < p < 1"
fun_expr(ψ::ShiftedProjLpBall) = "t ↦ χ({‖xk + sj + t‖ₚ ≤ r})"
fun_params(ψ::ShiftedProjLpBall) = "xk = $(ψ.xk)" * " "^14 * "sj = $(ψ.sj)"

"""
    shift!(ψ::ShiftedProjLpBall, shift::AbstractVector)

Updates the shift of a ShiftedNormLp object.
"""
function shift!(ψ::ShiftedProjLpBall, shift::AbstractVector{R}) where {R<:Real}
    if ψ.shifted_twice
        ψ.sj .= shift
    else
        ψ.xk .= shift
    end
    return ψ
end

# Allows ShiftedNormLp objects to be called as functions
function (ψ::ShiftedProjLpBall)(y::AbstractVector)
    @. ψ.xsy = ψ.xk + ψ.sj + y
    return ψ.h(ψ.xsy)
end

"""
    update_prox_context!(solver, stats, ψ::ShiftedProjLpBall)

Updates the context of a ShiftedProjLpBall object before calling prox!.

# Arguments
- `solver`: solver object
- `stats`: stats object
- `ψ`: ShiftedProjLpBall object
- `T`: Type of the object
"""
function update_prox_context!(solver, stats, ψ::ShiftedProjLpBall)
    ψ.h.context.hk = stats.solver_specific[:nonsmooth_obj]
    ψ.h.context.∇fk = solver.∇fk
    @. ψ.h.context.shift = ψ.xk + ψ.sj
end

"""
    prox!(y, h::ProjLpBall, q, ν)
    Evaluates inexactly the proximity operator of a Lp ball object.
    The duality gap at the solution is guaranteed to be less than `dualGap`.

    Inputs:
    - `y`: Array in which to store the result.
    - `h`: ProjLpBall object.
    - `q`: Vector to which the proximity operator is applied.
    - `ν`: Scaling factor.
"""
function prox!(y::AbstractArray, h::ProjLpBall, q::AbstractArray, ν::Real)
    ctx_projLpflag = h.context.flag_projLp
    h.context.flag_projLp = 1 # enforce regular IRBP stopping criterion : if prox! is called on h, the context is not created yet.
    x_irbp, dual_val, iters, _ =
        irbp_alg(q, h.p, h.radius, h.context, dualGap = h.context.dualGap, maxIter = 1000)
    y .= x_irbp
    # add the number of iterations in prox to the context object
    h.context.prox_stats[3] += iters
    h.context.flag_projLp = ctx_projLpflag # restore the original flag_projLp value
    return y
end

"""
    prox!(y, h::ShiftedProjLpBall, q, ν, ctx_ptr, callback)
    Evaluates inexactly the proximity operator of a shifted LpBall object.
    The duality gap at the solution is guaranteed to be less than `dualGap`.

    Inputs:
    - `y`: Array in which to store the result.
    - `ψ`: ShiftedProjLpBall object.
    - `q`: Vector to which the proximity operator is applied.
    - `ν`: Scaling factor.
"""
function prox!(y::AbstractArray, ψ::ShiftedProjLpBall, q::AbstractArray, ν::Real)
    context = ψ.h.context
    @. context.q_shifted = q + context.shift
    cond = false
    k = 0
    sum_iters = 0

    ψ.h.context.ν = ν

    while (cond == false) && (k ≤ context.iters_prox_projLp)
        # compute solution of the proximal operator of the shifted Lp ball
        x_irbp, dual_val, iters, _ = irbp_alg(
            context.q_shifted,
            ψ.h.p,
            ψ.h.radius,
            context,
            dualGap = context.dualGap,
            maxIter = 1000,
        )
        sum_iters += iters

        # compute model reduction
        context.s_k_unshifted .= x_irbp .- context.shift
        ϕk_val = dot(context.∇fk, context.s_k_unshifted)
        ψk_val = indicator_function(context.s_k_unshifted, ψ.h.p, ψ.h.radius)
        mk_val = ϕk_val + ψk_val
        ξk = context.hk - mk_val + max(1, abs(context.hk)) * 10 * eps()

        # update best solution
        if ξk > 0
            cond = true
        end
        k += 1
    end
    y .= context.s_k_unshifted
    # add the number of iterations in prox to the context object
    context.prox_stats[3] += sum_iters

    if cond == false
        @warn "Lp ball - prox computation could not find a feasible solution after $(context.iters_prox_projLp) runs."
    end
    return y
end
