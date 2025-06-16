using LinearAlgebra
using Printf
using IterativeSolvers

# Load matrix function (same as your original)
function load_matrix(system::String)
    if system == "He"
        N = 4488
    elseif system == "hBN"
        N = 6863        
    elseif system == "Si"
        N = 6201
    else
        error("Unknown system: $system")
    end

    # read the matrix
    filename = "../Davidson_algorithm/m_pp_" * system * ".dat"
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A  # for largest eigenvalues of original matrix
    return Hermitian(A)
end

# Jacobi-Davidson method
function jacobi_davidson(
    A::Hermitian{Float64,Matrix{Float64}},
    v0::Vector{Float64},
    m::Int,                # max subspace size before restart
    tol::Float64,
    system::String
)
    N = length(v0)
    v1 = v0 / norm(v0)
    w1 = A * v1
    h11 = dot(v1, w1)

    V = [v1]
    W = [w1]
    H = [h11]

    u = v1
    θ = h11
    r = w1 - θ * u

    iter = 0
    logfile = open("jd_log_$system.txt", "w")

    while norm(r) > tol
        iter += 1
        Vk = hcat(V...)
        Wk = hcat(W...)
        Hk = Matrix{Float64}(undef, size(Wk, 2), size(Vk, 2))

        # Inner loop
        for k = 1:m-1
            # Solve: (I - uu^*)(A - θI)(I - uu^*) t = -r approximately
            function jd_operator(t)
                t = t - u * dot(u, t) # (I - uu^*) t
                At = A * t - θ * t
                At = At - u * dot(u, At) # (I - uu^*) (A - θI) (I - uu^*) t
                return At
            end

            t = zeros(N)
            maxit = 10
            tol_lin = 1e-3
            t, _ = gmres(jd_operator, -r, restart=10, maxiter=maxit, tol=tol_lin, log=false)

            # Orthonormalize t against Vk
            for vi in V
                t -= vi * dot(vi, t)
            end
            t /= norm(t)

            push!(V, t)
            wk1 = A * t
            push!(W, wk1)
        end

        Vk = hcat(V...)
        Wk = hcat(W...)
        Hk = Vk' * Wk

        # Compute largest eigenpair
        Σ, S = eigen(Hermitian(Hk))
        θ, idx = findmax(real(Σ))
        s = S[:, idx]
        u = Vk * s
        û = Wk * s
        r = û - θ * u

        println("iter=$iter  |r| = $(norm(r))")
        @printf(logfile, "%d %.6e\n", iter, norm(r))

        # Restart with best vector
        V = [u]
        W = [û]
        H = [θ]
    end

    close(logfile)
    return θ, u
end

# Main function
function main(system::String)
    A = load_matrix(system)
    N = size(A, 1)

    # Initial guess vector (random)
    v0 = randn(N)
    
    println("Running Jacobi-Davidson for system: $system")
    @time θ, u = jacobi_davidson(A, v0, 20, 1e-5, system)

    println("\nApproximate dominant eigenvalue: ", θ)
end

# Example usage
main("Si")
