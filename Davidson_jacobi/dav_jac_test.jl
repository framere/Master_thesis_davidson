using LinearAlgebra
using IterativeSolvers

# -----------------------------
# Struct to hold JD parameters
# -----------------------------
Base.@kwdef struct JDOptions
    k_max::Int
    m_max::Int
    θ_target::Float64
    tol::Float64
end


# -------------------------------------------------------
# Solve the Jacobi–Davidson correction equation
# for standard eigenvalue problems: (A - θ I) x = λ x
# -------------------------------------------------------

function solve_correction_equation_standard(A, θ, u, r, Q̃)
    Π = I - Q̃ * Q̃'
    Ā = Π * (A - θ * I) * Π
    Ā_sym = Symmetric(Matrix(Ā))  # Ensure CG gets a symmetric matrix

    t = zeros(size(A, 1))
    cg!(t, Ā_sym, -r; reltol=1e-2, log=false)

    return t
end


# -------------------------------------------------------
# Jacobi-Davidson algorithm for standard eigenvalue problems
# -------------------------------------------------------
function jacobi_davidson_standard(A::AbstractMatrix, opts::JDOptions)
    n = size(A, 1)
    V = Matrix{Float64}(undef, n, 0)   # Search subspace
    Q = Matrix{Float64}(undef, n, 0)   # Deflation space
    M = zeros(0, 0)                    # Projected matrix
    k = 0
    m = 0
    t = randn(n)
    t /= norm(t)

    while k < opts.k_max
        # Orthonormalize t against current V
        for i in 1:m
            t -= (V[:, i]' * t) * V[:, i]
        end
        t /= norm(t)

        # Expand subspace
        V = hcat(V, t)
        vA = A * t
        m += 1

        # Expand projected matrix M
        if size(M, 1) < m
            M = [M zeros(m-1); zeros(1, m)]
        end
        for i in 1:m
            M[i, m] = V[:, i]' * vA
            M[m, i] = M[i, m]
        end

        # Solve projected eigenproblem
        evals, evecs = eigen(Symmetric(M))
        idx = argmin(abs.(evals .- opts.θ_target))
        θ = evals[idx]
        s = evecs[:, idx]

        # Approximate eigenvector and residual
        u = V * s
        r = A * u - θ * u

        println("Iter $k: θ = $θ, ‖r‖ = $(norm(r))")
        if norm(r) < opts.tol
            println("✅ Converged eigenvalue: θ = $θ")
            return θ, u
        end

        # Solve correction equation for next direction
        Q̃ = hcat(Q, u)
        t = solve_correction_equation_standard(A, θ, u, r, Q̃)
        k += 1
    end

    error("❌ Jacobi–Davidson did not converge in $(opts.k_max) iterations.")
end

# -----------------------------
# Example usage
# -----------------------------

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


function main(system::String)
    A = load_matrix(system)

    # Set options: target is largest eigenvalue
    opts = JDOptions(k_max=50, m_max=20, θ_target=5.0, tol=1e-3)

    # Run Jacobi–Davidson
    θ, u = jacobi_davidson_standard(A, opts)

    println("Approximate eigenvalue: θ = ", θ)
    println("Residual norm: ", norm(A * u - θ * u))

    # exact eigenvalue for comparison
    println("Perform exact diagonalization for reference")
    @time exact_eigenvals, exact_eigenvecs = eigen(A)
    exact_θ = maximum(- exact_eigenvals)
    println("Exact eigenvalue: θ_exact = ", exact_θ)
end

main("He")  # Change to "hBN" or "Si" as needed
