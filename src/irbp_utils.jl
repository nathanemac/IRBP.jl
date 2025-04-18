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
function pnorm(x::AbstractVector{T}, p::Real) where {T<:Real}
    return (sum(abs.(x) .^ p))^(1 / p)
end


"""
    get_hyperplane_projection(x, weights, radius)
    Compute the projection of `x` on the hyperplan defined by `weights`.
"""
function get_hyperplane_projection(x, weights, radius)
    eps_val = eps(Float64)
    numerator = dot(weights, x) - radius
    denominator = dot(weights, weights)
    dual = numerator / (denominator + eps_val)
    x_sub = x .- dual .* weights
    return x_sub, dual
end


"""
    get_weightedl1_ball_projection(point_to_be_projected, weights, radius)
    Compute projection of a vector `point_to_be_projected`
    on the weighted ℓ1 ball of radius `radius`.

    Returns:
    - x_opt: the projected point
    - dual: the associated dual value
"""
function get_weightedl1_ball_projection(point_to_be_projected, weights, radius)

    signum_vals = sign.(point_to_be_projected)

    point_to_be_projected_copy = abs.(point_to_be_projected)

    act_ind = trues(length(point_to_be_projected))

    while true
        point_to_be_projected_copy_act = point_to_be_projected_copy[act_ind]
        weights_act = weights[act_ind]

        x_sol_hyper, dual =
            get_hyperplane_projection(point_to_be_projected_copy_act, weights_act, radius)

        # On remet à zéro les valeurs négatives
        x_sol_hyper_clamped = max.(x_sol_hyper, 0.0)
        point_to_be_projected_copy_act .= x_sol_hyper_clamped

        # Mise à jour des valeurs dans le vecteur complet
        point_to_be_projected_copy[act_ind] .= x_sol_hyper_clamped

        # Mise à jour de l'ensemble actif
        act_ind .= point_to_be_projected_copy .> 0.0

        # Comptage des inactifs (ceux devenus < 0 dans x_sol_hyper)
        inact_ind_cardinality = sum(x_sol_hyper .< 0.0)

        # Si plus aucun élément n'est supprimé à cette itération, on s'arrête
        if inact_ind_cardinality == 0
            # Restaure le signe initial
            x_opt = point_to_be_projected_copy .* signum_vals
            return x_opt, dual
        end
    end
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

"""
    ModelFunction{V,P}

Helper structure to store the gradient and proximal term in a compact way.
Used to avoid memory allocations when calling the proximal callback.

# Fields
- `∇f::V`: gradient of the function
- `ψ::P`: proximal term
"""
mutable struct ModelFunction{V,P}
    ∇f::V  # gradient
    ψ::P   # proximal term
end

"""
    ModelFunction(∇f::V, ψ::Function) where {V<:AbstractVector}

Constructor for ModelFunction that creates a structure with a gradient vector and a proximal function.
"""
function ModelFunction(∇f::V, ψ::Function) where {V<:AbstractVector}
    return ModelFunction{V,Function}(∇f, ψ)
end

"""
    (m::ModelFunction)(d)

Evaluate the model function at point d by computing the sum of:
1. The inner product between the gradient and d
2. The proximal term evaluated at d
"""
function (m::ModelFunction)(d)
    return dot(m.∇f, d) + m.ψ(d)
end

mutable struct IRBPContext
    iters_prox_projLp::Int64
    flag_projLp::Int64
    κξ::Float64
    dualGap::Float64
    prox_stats::Vector{Int64}
    shift::Vector{Float64}
    s_k_unshifted::Vector{Float64}
    hk::Float64
    mk::ModelFunction
end

# Fonction that creates an IRBPContext object
function IRBPContext(
    n::Int64;
    iters_prox_projLp = 100,
    flag_projLp = 0,
    κξ = 0.75,
    dualGap = 1e-8,
)
    shift = zeros(n)
    s_k_unshifted = zeros(n)
    hk = 0.0
    mk = ModelFunction(zeros(n), x -> x)
    prox_stats = zeros(Int64, 3)
    return IRBPContext(
        iters_prox_projLp,
        flag_projLp,
        κξ,
        dualGap,
        prox_stats,
        shift,
        s_k_unshifted,
        hk,
        mk,
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
    (h::ProjLpBall)(x::AbstractVector)
    Indicator function for the p-ball.
    Returns zero if the point is inside the ball, Inf otherwise.
    A small ϵ is added to the radius to avoid numerical issues.
    This function is the "h" function in the RegularizedOptimization.jl framework.
"""
function (h::ProjLpBall)(x::AbstractVector; ϵ::Real = eps()^(1 / 2))
    pnorm(x, h.p)^(h.p) <= (h.radius + ϵ) ? 0.0 : Inf
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
    ψ.h.context.mk.∇f = solver.∇fk
    ψ.h.context.mk.ψ = d -> ψ(d)  # Use the evaluation function of ψ instead of the object itself
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
    q_shifted = q .+ context.shift
    cond = false
    k = 0
    sum_iters = 0

    while (cond == false) && (k ≤ context.iters_prox_projLp)
        # compute solution of the proximal operator of the shifted Lp ball
        x_irbp, dual_val, iters, _ = irbp_alg(
            q_shifted,
            ψ.h.p,
            ψ.h.radius,
            context,
            dualGap = context.dualGap,
            maxIter = 1000,
        )
        sum_iters += iters

        # compute model reduction
        context.s_k_unshifted .= x_irbp .- context.shift
        ξk =
            context.hk - context.mk(context.s_k_unshifted) +
            max(1, abs(context.hk)) * 10 * eps()

        # update best solution
        if ξk > 0
            cond = true
        end
        k += 1
    end
    y .= context.s_k_unshifted
    # add the number of iterations in prox to the context object
    context.prox_stats[3] += sum_iters

    cond == false ?
    println(
        "Warning: Lp ball - prox computation could not find a feasible solution after $(context.iters_prox_projLp) runs.",
    ) : nothing # println("Lp ball - prox computation found a feasible solution : ξk = $best_ξk")
    return y
end
