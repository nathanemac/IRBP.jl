include("irbp_utils.jl")
include("irbp_alg.jl")
using Plots
# -----------------------------------------
# 1. Fonction de tracé de la boule L1 pondérée
# -----------------------------------------
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

# -------------------------
# 2. Exemple d'utilisation
# -------------------------
weights = [1.5, 0.5]        # Poids sur chaque composante
radius = 1.0               # Rayon de la boule L1 pondérée
point = [-2.0, 2.0]       # Point à projeter

# Projection
x_opt, dual_val = get_weightedl1_ball_projection(point, weights, radius)

# Création de la figure
plt = plot_weightedl1_ball_2D(weights, radius; color = :blue)

# Ajout du point initial (en rouge)
scatter!(plt, [point[1]], [point[2]], color = :red, label = "Point à projeter")

# Ajout du point projeté (en vert)
scatter!(plt, [x_opt[1]], [x_opt[2]], color = :green, label = "Point projeté")

# Affichage du résultat
display(plt)


##########################################
# ==================================================================
# 5) plot_lp_ball_2D : pseudo-boule pour p < 1 ou vraie boule p >= 1
# ==================================================================
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

# ==================================================================
# EXEMPLE COMPLET
# ==================================================================

p = 0.5
radius = 2.0
point_original = [-4.0, -2.0]

# Lance la projection IRBP (qui renvoie aussi x_list)
point_projected, dual, iters, elapsed, x_list =
    IRBP(point_original, p, radius, dualGap = 1e-4)

# Trace de la boule p
plt = plot_lp_ball_2D(p, radius; color = :blue, npoints = 1000)

# Ajout du point initial
scatter!(
    plt,
    [point_original[1]],
    [point_original[2]],
    color = :red,
    label = "Original point",
)

# Ajout de tous les itérés (ligne noire + marqueurs)
# x_list est un Vector{Vector{Float64}}, on va extraire x et y
xs = [v[1] for v in x_list]
ys = [v[2] for v in x_list]

plot!(
    plt,
    xs,
    ys,
    marker = :o,
    line = :dash,
    color = :black,
    label = "IRBP iterates",
    series_annotations = 1:length(x_list),
)

# Ajout du point final projeté (en vert)
scatter!(
    plt,
    [point_projected[1]],
    [point_projected[2]],
    color = :green,
    label = "Projected point",
)

title!(plt, "IRBP with p=$p in (0,1), radius=$radius")
xlabel!(plt, "x1")
ylabel!(plt, "x2")

display(plt)


##########################################
# Tests de prox!
##########################################

h = ProjLpBall(1.0, 0.5, 2.0)
q = [3.0, 2.0]
y = similar(q)
ν = 0.5
context = AlgorithmContextCallback(dualGap = 1e-4)
callback = @cfunction(x -> nothing, Cvoid, (Ptr{Cvoid},))
prox!(y, h, q, ν, context, callback)
