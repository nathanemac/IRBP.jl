# This file contains utility functions used in the package.
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
