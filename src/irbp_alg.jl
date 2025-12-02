"""
    get_hyperplane_projection!(x, w, m, radius, s_sub) -> (dual)
    Compute the projection of `x` on the hyperplan defined by `weights` and stores the result in `s_sub`.
"""
@inline function get_hyperplane_projection!(
    x::AbstractVector{T},
    w::AbstractVector{T},
    m::Int,
    radius::T,
    s_sub::AbstractVector{T},
) where {T<:Float64}

    # -- dual ---------------------------------------------------------------
    num = zero(T)
    den = zero(T)
    @inbounds @simd for i = 1:m
        wi = w[i]
        num += wi * x[i]
        den += wi * wi
    end
    dual = (num - radius) / (den + eps(T))

    # -- projection ----------------------------------------------
    @inbounds @simd for i = 1:m
        s_sub[i] = x[i] - dual * w[i]
    end
    return dual
end


"""
    get_weightedl1_ball_projection(point_to_be_projected, weights, radius)
    Compute projection of a vector `point_to_be_projected`
    on the weighted ℓ1 ball of radius `radius`.

    Returns:
    - x_opt: the projected point
    - dual: the associated dual value
"""
function get_weightedl1_ball_projection(context, radius)
    y, n = context.yAbs, length(context.yAbs)

    # -------- init ---------------------------------------------------------
    @inbounds @simd for i = 1:n
        v = y[i]
        context.signum_vals_l1[i] = copysign(1.0, v)
        context.point_to_be_projected_l1[i] = abs(v)
        context.act_ind_l1[i] = true
    end

    while true
        # -------- gather ---------------------------------------------------
        m = 0
        @inbounds for i = 1:n
            if context.act_ind_l1[i]
                m += 1
                context.point_to_be_projected_act_l1[m] =
                    context.point_to_be_projected_l1[i]
                context.weights_act_l1[m] = context.weights[i]
            end
        end

        # -------- projection ----------------------------------------------
        dual = get_hyperplane_projection!(
            context.point_to_be_projected_act_l1,
            context.weights_act_l1,
            m,
            radius,
            context.s_sub,
        )

        # -------- clamp + scatter -----------------------------------------
        removed = false
        idx = 0
        @inbounds for i = 1:n
            if context.act_ind_l1[i]
                idx += 1
                val = context.s_sub[idx]      # <-- on lit la projection !
                if val < 0.0
                    val = 0.0
                    removed = true
                end
                context.point_to_be_projected_l1[i] = val
                context.act_ind_l1[i] = val > 0.0
            end
        end

        # -------- arrêt ----------------------------------------------------
        if !removed
            @inbounds @simd for i = 1:n
                context.x_opt_l1[i] =
                    context.point_to_be_projected_l1[i] * context.signum_vals_l1[i]
            end
            return dual
        end
    end
end


"""
    get_lp_ball_projection(starting_point, point_to_be_projected, p, radius, epsilon;
                           tau=1.1, tol=1e-8, MAX_ITER=1000)

This function projects a point `point_to_be_projected` onto the lp-ball of radius `radius` using
the IRBP (Iterative Reweighted Balls Projection) method. It takes an initial guess `starting_point`
and a smoothing vector `epsilon`, then iterates until convergence or until `MAX_ITER` is reached.

Arguments:
  - starting_point      : The initial iterate for IRBP.
  - point_to_be_projected : The point to be projected (Vector).
  - p                   : The p-parameter for the lp-ball.
  - radius             : The radius of the lp-ball.
  - epsilon            : Initial smoothing parameter vector for IRBP.
  - tau                : Parameter for updating the smoothing condition (default=1.1).
  - tol                : Tolerance for stopping criterion (default=1e-8).
  - MAX_ITER           : Maximum number of iterations (default=1000).

Returns:
  - x_final    : The projection of `point_to_be_projected` onto the lp-ball.
  - dual       : The final dual variable.
  - cnt       : Number of iterations.
  - runningTime : Time in seconds spent in the IRBP loop.
"""
function get_lp_ball_projection(
    starting_point::Vector{Float64},
    point_to_be_projected::Vector{Float64},
    p::Float64,
    radius::Float64,
    epsilon::Vector{Float64},
    context::IRBPContext;
    tau::Float64 = 1.1,
    tol::Float64 = 1e-8,
    MAX_ITER::Int = 1000,
)

    # If the point is already inside the lp-ball (norm^p <= radius), return immediately
    if pnorm(point_to_be_projected, p)^p <= radius
        return point_to_be_projected, 0.0, 0, 0.0
    end

    # plt = plot_lp_ball_2D(p, radius; color = :blue, npoints = 5000)
    # scatter!(
    #     plt,
    #     [point_to_be_projected[1]],
    #     [point_to_be_projected[2]],
    #     color = :red,
    #     label = "Original point ($(point_to_be_projected[1]), $(point_to_be_projected[2]))"
    #     )

    # Problem dimension
    n = Float64(length(point_to_be_projected))

    # Constant threshold used to compare with 'condition_left' (M in the paper)
    condition_right = 100.0

    # 'signum' extracts signs of each component
    @. context.signum_vals = sign.(point_to_be_projected)

    # yAbs will be the positive version of point_to_be_projected
    # taking into account the sign of each component
    @. context.yAbs = context.signum_vals .* point_to_be_projected

    # Initialize the dual variable
    lamb = 0.0

    # Initial residuals for alpha and beta
    @. context.temp_vec = (context.yAbs - context.x_ini) * context.x_ini
    residual_alpha0 = (1.0 / n) * pnorm(context.temp_vec, 1.0)
    residual_beta0 = abs(pnorm(context.x_ini, p)^p - radius)

    # Counter for iterations
    cnt = 0

    alpha_res = Inf
    beta_res = Inf
    ξk = Inf

    # Start measuring time
    timeStart = time()

    while cnt < MAX_ITER
        cnt += 1

        # Compute current residuals
        @. context.temp_vec =
            (context.yAbs - context.x_ini) * context.x_ini - p * lamb * (context.x_ini^p)
        alpha_res = (1.0 / n) * pnorm(context.temp_vec, 1.0)
        beta_res = abs(pnorm(context.x_ini, p)^p - radius)

        if context.flag_projLp == 1 # original IRBP criterion
            if max(alpha_res, beta_res) <
               tol * max(max(residual_alpha0, residual_beta0), 1.0) || cnt > MAX_ITER
                timeEnd = time()
                x_final = context.signum_vals .* context.x_ini  # Restore original sign
                return x_final, lamb, cnt, (timeEnd - timeStart)
            end
        else # stopping condition that respects our assumptions on inexact prox computation
            delta_k = max(alpha_res, beta_res)
            @. context.s_k = context.signum_vals * context.x_ini

            @. context.s_k_unshifted = context.s_k - context.shift
            ϕk_val = dot(context.∇fk, context.s_k_unshifted)
            ψk_val = indicator_function(context.s_k_unshifted, context.p, context.radius)
            mk_val = ϕk_val + ψk_val
            ξk = context.hk - mk_val + max(1, abs(context.hk)) * 10 * eps()

            s_norm = norm(context.s_k_unshifted)
            bound_s = norm(context.shift) + context.radius^(1 / context.p)

            condition =
                (
                    s_norm ≥ context.κs * bound_s ||
                    delta_k ≤ tol * max(max(residual_alpha0, residual_beta0), 1.0)
                ) && ξk > 0
            if condition
                timeEnd = time()
                return context.s_k, lamb, cnt, (timeEnd - timeStart)
            end
        end



        # Step 3 in IRBP: compute the weights
        # weights_i = p / (|x_i| + epsilon)^(1-p)
        # Add 1e-12 to avoid division by zero in the denominator
        @. context.temp_vec = abs(context.x_ini) + epsilon
        @. context.weights = p * (1.0 ./ ((context.temp_vec) .^ (1.0 - p) .+ 1e-12))

        # Step 4 in IRBP: compute gamma_k
        # gamma_k = radius - |||x| + epsilon||_p^p + sum(weights .* |x|)
        gamma_k1 = radius - (pnorm(context.temp_vec, p)^p)
        @. context.temp_vec -= epsilon
        gamma_k2 = dot(context.weights, context.temp_vec)
        gamma_k = gamma_k1 + gamma_k2

        @assert gamma_k > 0 "The current Gamma is non-positive"

        # Subproblem solver:
        # Weighted L1-ball projection of yAbs with weights and gamma_k
        # the result is stored in context.x_opt_l1
        lamb = get_weightedl1_ball_projection(context, gamma_k)

        # Replace any NaN values by zero (if any)
        @inbounds @simd for i in eachindex(context.x_opt_l1)
            if isnan(context.x_opt_l1[i])
                context.x_opt_l1[i] = 0.0
            end
        end

        # Step 5 in IRBP: update epsilon if condition_left <= condition_right
        @. context.temp_vec = context.x_opt_l1 - context.x_ini
        norm_aux1 = pnorm(context.temp_vec, 2.0)
        @. context.temp_vec = sign(context.temp_vec) * context.weights
        norm_aux2 = pnorm(context.temp_vec, 2.0)
        condition_left = (norm_aux1) * (norm_aux2)^tau

        if condition_left <= condition_right
            # Update factor for epsilon
            theta = (min(beta_res, 1.0 / sqrt(cnt)))^(1.0 / p)
            epsilon .= epsilon .* theta
        end

        # Step 6 in IRBP: update the iterate
        context.x_ini .= context.x_opt_l1

        # if cnt % 10 == 0
        #     x_signed = signum_vals .* context.x_ini
        #     scatter!(
        #         plt,
        #         [x_signed[1]],
        #         [x_signed[2]],
        #         color = :green,
        #         label = "iterate at $(cnt) : ($(x_signed[1]), $(x_signed[2]))"
        #         )
        # end
    end

    # Final result: restore the original sign
    timeEnd = time()
    @. context.s_k = context.signum_vals * context.x_ini


    # scatter!(
    #     plt,
    #     [x_final[1]],
    #     [x_final[2]],
    #     color = :green,
    #     label = "Final point ($(x_final[1]), $(x_final[2]))"
    #     )
    # display(plt)
    return context.s_k, lamb, cnt, (timeEnd - timeStart)
end

"""
    irbp_alg(
    point_to_be_projected::Vector{Float64},
    p::Float64,
    radius::Float64;
    dualGap = 1e-8,
    maxIter = 1000,
)

This function initializes an IRBP iteration for projecting a point onto the p-ball of radius `radius`.
It returns the projected point `x_irbp`, the dual variable `dual`, and the total running time.

Arguments:
  - point_to_be_projected : The point to be projected (Vector).
  - p                     : The p-parameter for the lp-ball.
  - radius               : The radius of the lp-ball.

Returns:
  - x_irbp    : The projection of the point onto the lp-ball.
  - dual      : The dual variable from the IRBP solver.
  - runningTime : Time elapsed (in seconds) during the IRBP process.
"""
function irbp_alg(
    point_to_be_projected::Vector{Float64},
    p::Float64,
    radius::Float64,
    context::IRBPContext;
    dualGap = 1e-8,
    maxIter = 1000,
)
    # Dimension of the data
    data_dim = length(point_to_be_projected)

    # Create a random vector in [0,1] for initialization
    @. context.rand_num = rand()
    rand_num_norm = pnorm(context.rand_num, 1.0)

    # Slightly shrink the random vector to ensure feasibility (according to paper)
    # (raised to the power 1/p)
    @inbounds for i in eachindex(context.rand_num)
        context.epsilon_ini[i] =
            0.9 * (context.rand_num[i] * radius / rand_num_norm)^(1.0 / p)
    end

    context.x_ini .= 0.0

    # Call the IRBP-based projection on lp-ball
    x_irbp, dual, iters, runningTime = get_lp_ball_projection(
        context.x_ini,
        point_to_be_projected,
        p,
        radius,
        context.epsilon_ini,
        context;
        tau = 1.1,
        tol = dualGap,
        MAX_ITER = maxIter,
    )

    return x_irbp, dual, iters, runningTime
end
