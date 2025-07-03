using LinearAlgebra
using Printf

include("../uptodate_codes/functions_davidson.jl")

# === Global FLOP counter and helpers ===
global NFLOPs = 0

function count_matmul_flops(M::Int, N::Int, K::Int)
    global NFLOPs += 2 * M * N * K
end

function count_diag_flops(N::Int)
    global NFLOPs += 20 * N^3
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    system::String = "",
    stable_thresh::Integer = 6
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    global NFLOPs

    n_b = size(V, 2)
    nu_0 = max(l, n_b)
    nevf = 0

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)

    iter = 0
    convergence_tracker = Dict{Int, Tuple{Float64, Int, Float64, Vector{T}}}()

    while nevf < l
        iter += 1

        if size(V_lock, 2) > 0
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    V[:, j] -= v_lock * (v_lock' * V[:, j])
                end
            end
        end
        V = Matrix(qr(V).Q)

        # Rayleigh-Ritz
        AV = A * V
        count_matmul_flops(size(A, 1), size(A, 2), size(V, 2))
        H = Hermitian(V' * AV)
        count_matmul_flops(size(V, 2), size(V, 1), size(AV, 2))

        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)
        count_diag_flops(size(H, 1))

        X = V * U
        count_matmul_flops(size(V, 1), size(V, 2), size(U, 2))

        R = X .* Σ' .- A * X
        count_matmul_flops(size(A, 1), size(A, 2), size(X, 2))

        norms = vec(norm.(eachcol(R)))

        conv_indices = Int[]
        for i = 1:size(R, 2)
            λ = Σ[i]
            rnorm = norms[i]

            if haskey(convergence_tracker, i)
                λ_prev, count, _, _ = convergence_tracker[i]
                if abs(λ - λ_prev) < 1e-6 && rnorm < thresh
                    convergence_tracker[i] = (λ, count + 1, rnorm, X[:, i])
                else
                    convergence_tracker[i] = (λ, 1, rnorm, X[:, i])
                end
            elseif rnorm < thresh
                convergence_tracker[i] = (λ, 1, rnorm, X[:, i])
            end

            if haskey(convergence_tracker, i)
                λ, count, rnorm, xvec = convergence_tracker[i]
                if count >= stable_thresh
                    push!(conv_indices, i)
                    push!(Eigenvalues, λ)
                    Ritz_vecs = hcat(Ritz_vecs, xvec)
                    V_lock = hcat(V_lock, xvec)
                    delete!(convergence_tracker, i)
                    nevf += 1
                    println(@sprintf("EV %3d converged λ = %.10f, ‖r‖ = %.2e, stable for %d iters", nevf, λ, rnorm, count))
                    if nevf >= l
                        println("Converged all eigenvalues.")
                        return (Eigenvalues, Ritz_vecs)
                    end
                end
            end
        end

        non_conv_indices = setdiff(1:size(R, 2), conv_indices)
        X_nc = X[:, non_conv_indices]
        Σ_nc = Σ[non_conv_indices]
        R_nc = R[:, non_conv_indices]

        t = Matrix{T}(undef, size(A, 1), length(non_conv_indices))
        ϵ = 1e-6
        for (i, idx) in enumerate(non_conv_indices)
            denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
            t[:, i] = R_nc[:, i] ./ denom
        end

        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-10)

        if size(V, 2) + n_b_hat > n_aux || length(conv_indices) > 0 || n_b_hat == 0
            V = hcat(X_nc, T_hat)
            n_b = size(V, 2)
        else
            V = hcat(V, T_hat)
            n_b += n_b_hat
        end

        println("Iter $iter: V_size = $n_b, Converged = $nevf, Min ‖r‖ = $(minimum(norms))")
    end

    return (Eigenvalues, Ritz_vecs)
end

function main(system::String, l::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../Davidson_algorithm/m_pp_" * system * ".dat"

    Nlow = 16
    Naux = Nlow * 16

    A = load_matrix(system, filename)
    N = size(A, 1)

    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i, i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, l, 1e-2, system, 3)

    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]

    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A)
    count_diag_flops(N)

    display("text/plain", (Σ - Σexact[1:l])')
    println("Total estimated FLOPs: $(NFLOPs)")
end

ls = [216, 288, 360]

for l in ls
    println("Running for Nlow = $l")
    main("hBN", l)
end
