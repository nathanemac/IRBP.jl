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
    radius::Float64;
    dualGap = 1e-8,
    maxIter = 1000,
)
    # Dimension of the data
    data_dim = length(point_to_be_projected)

    # Initialize starting point with zeros
    x_ini = zeros(Float64, data_dim)

    # Create a random vector in [0,1] for initialization
    rand_num = rand(Float64, data_dim)
    rand_num_norm = norm(rand_num, 1)

    # Slightly shrink the random vector to ensure feasibility (according to paper)
    # (raised to the power 1/p)
    epsilon_ini = 0.9 .* (rand_num .* radius ./ rand_num_norm) .^ (1.0 / p)

    # Call the IRBP-based projection on lp-ball
    x_irbp, dual, iters, runningTime, x_list = get_lp_ball_projection(
        x_ini,
        point_to_be_projected,
        p,
        radius,
        epsilon_ini;
        tau = 1.1,
        tol = dualGap,
        MAX_ITER = maxIter,
    )

    return x_irbp, dual, iters, runningTime, x_list
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
    epsilon::Vector{Float64};
    tau::Float64 = 1.1,
    tol::Float64 = 1e-8,
    MAX_ITER::Int = 1000,
)

    # If the point is already inside the lp-ball (norm^p <= radius), return immediately
    if pnorm(point_to_be_projected, p)^p <= radius
        return point_to_be_projected, 0.0, 0, 0.0, [point_to_be_projected]
    end

    # Problem dimension
    n = length(point_to_be_projected)

    # Constant threshold used to compare with 'condition_left' (M in the paper)
    condition_right = 100.0

    # 'signum' extracts signs of each component
    signum_vals = sign.(point_to_be_projected)

    # yAbs will be the positive version of point_to_be_projected
    # taking into account the sign of each component
    yAbs = signum_vals .* point_to_be_projected

    # Initialize the dual variable
    lamb = 0.0

    # Initial residuals for alpha and beta
    residual_alpha0 =
        (1.0 / n) * norm(
            (yAbs .- starting_point) .* starting_point .- p * lamb .* (starting_point .^ p),
            1,
        )
    residual_beta0 = abs(pnorm(starting_point, p)^p - radius)

    # Counter for iterations
    cnt = 0

    x_list = Vector{Float64}[]
    push!(x_list, starting_point)

    # Start measuring time
    timeStart = time()

    while cnt < MAX_ITER
        cnt += 1

        # Compute current residuals
        alpha_res =
            (1.0 / n) * norm(
                (yAbs .- starting_point) .* starting_point .-
                p * lamb .* (starting_point .^ p),
                1,
            )
        beta_res = abs(pnorm(starting_point, p)^p - radius)

        # Stopping criterion
        if max(alpha_res, beta_res) <
           tol * max(max(residual_alpha0, residual_beta0), 1.0) || cnt > MAX_ITER
            timeEnd = time()
            x_final = signum_vals .* starting_point  # Restore original sign
            push!(x_list, x_final)
            return x_final, lamb, cnt, (timeEnd - timeStart), x_list
        end

        # Step 3 in IRBP: compute the weights
        # weights_i = p / (|x_i| + epsilon)^(1-p)
        # Add 1e-12 to avoid division by zero in the denominator
        weights = p .* (1.0 ./ ((abs.(starting_point) .+ epsilon) .^ (1.0 - p) .+ 1e-12))

        # Step 4 in IRBP: compute gamma_k
        # gamma_k = radius - |||x| + epsilon||_p^p + sum(weights .* |x|)
        gamma_k =
            radius - (pnorm(abs.(starting_point) .+ epsilon, p)^p) +
            dot(weights, abs.(starting_point))

        @assert gamma_k > 0 "The current Gamma is non-positive"

        # Subproblem solver:
        # Weighted L1-ball projection of yAbs with weights and gamma_k
        x_new, lamb = get_weightedl1_ball_projection(yAbs, weights, gamma_k)

        # Replace any NaN values by zero (if any)
        x_new[isnan.(x_new)] .= 0.0

        # Step 5 in IRBP: update epsilon if condition_left <= condition_right
        condition_left =
            norm(x_new .- starting_point, 2) *
            (norm(sign.(x_new .- starting_point) .* weights, 2)^tau)

        if condition_left <= condition_right
            # Update factor for epsilon
            theta = (min(beta_res, 1.0 / sqrt(cnt)))^(1.0 / p)
            epsilon .= epsilon .* theta
        end

        # Step 6 in IRBP: update the iterate
        starting_point = copy(x_new)
        push!(x_list, starting_point)
    end
end
