using LinearAlgebra
using Printf

include("functions_davidson.jl")


function davidson_driver(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Int,
    l::Int,
    thresh::Float64,
    n_final::Int
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    n_b = size(V, 2)
    nu_0 = max(l, n_b)
    nevf = 0

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)

    iter = 0

    while nevf < l
        iter += 1
        remaining = l - nevf

        if remaining <= n_final
            println("-----Switching to final convergence mode-----")
            while true
                iter += 1

                V = orthogonalize(V, V_lock)

                Σ, X, R = rayleigh_ritz_projection(A, V, remaining)
                Rnorm = norm(R, 2)  # Frobenius norm

                output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V, 2))
                print(output)

                if Rnorm < thresh
                    println("Converged!")
                    for i = 1:remaining
                        push!(Eigenvalues, Σ[i])
                        Ritz_vecs = hcat(Ritz_vecs, X[:, i])
                        V_lock = hcat(V_lock, X[:, i])
                        nevf += 1
                        println(@sprintf("Converged eigenvalue %.10f with norm %.2e (EV %d)", Σ[i], Rnorm, nevf))
                    end
                end
                
                t = zero(similar(R))
                for i in 1:size(t, 2)
                    ri = R[:,i]
                    vi = X[:,i]
                    λi = Σ[i]

                    # Projector: P = I - vi*vi'
                    Pi = I - vi * vi'

                    # Approximate (I - vi*vi') (A - λi I)^(-1) (I - vi*vi') * ri
                    M_diag_inv = 1.0 ./ (D .- λi)      # Diagonal preconditioner
                    zi = M_diag_inv .* (Pi * ri)      # Apply preconditioner to projected residual
                    si = Pi * zi                      # Project again to stay orthogonal to vi

                    t[:,i] = si
                end

                # # Update guess space using diagonal preconditioner
                # t = zero(similar(R))
                # for i = 1:size(t, 2)
                #     C = 1.0 ./ (Σ[i] .- D)
                #     t[:, i] = C .* R[:, i]  # the new basis vectors
                # end

                # Update guess basis
                if size(V, 2) <= Naux - remaining
                    V = hcat(V, t)  # concatenate V and t
                else
                    V = hcat(X, t)  # concatenate X and t
                end
                # Check if we have converged all eigenvalues
                if nevf >= l
                    println("Converged all eigenvalues.")
                    return Eigenvalues, Ritz_vecs
                end
            end 
        
        else
            V = orthogonalize(V, V_lock)

            # Rayleigh-Ritz

            nu = min(size(V, 2), nu_0 - nevf)
            Σ, X, R = rayleigh_ritz_projection(A, V, nu)
            norms = vec(norm.(eachcol(R)))

            conv_indices = Int[]
            for i = 1:size(R, 2)
                if norms[i] <= thresh
                    push!(conv_indices, i)
                    push!(Eigenvalues, Σ[i])
                    Ritz_vecs = hcat(Ritz_vecs, X[:, i])
                    V_lock = hcat(V_lock, X[:, i])
                    nevf += 1
                    println(@sprintf("Converged eigenvalue %.10f with norm %.2e (EV %d)", Σ[i], norms[i], nevf))
                    if nevf >= l
                        println("Converged all eigenvalues.")
                        return (Eigenvalues, Ritz_vecs)
                    end
                end
            end

            non_conv_indices = setdiff(1:size(R, 2), conv_indices)
            X_nc = X[:, non_conv_indices]
            Σ_nc = Σ[non_conv_indices]
            R_nc = R[:, non_conv_indices]

            # Correction vectors
            t = Matrix{T}(undef, size(A, 1), length(non_conv_indices))
            ϵ = 1e-6  # small value to avoid division by zero
            for (i, idx) in enumerate(non_conv_indices)
                denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
                t[:, i] = R_nc[:, i] ./ denom
            end

            # Orthogonalize corrections
            T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-10)
            # println("→ Requested $(size(t, 2)) corrections, kept $n_b_hat after orthogonalization")

            # Update subspace
            if size(V, 2) + n_b_hat > Naux|| length(conv_indices) > 0 || n_b_hat == 0 
                V = hcat(X_nc, T_hat)
                n_b = size(V, 2)
            else
                V = hcat(V, T_hat)
                n_b += n_b_hat
            end
            println("Iter $iter: V_size = $n_b, Converged = $nevf, Norm = $(minimum(norms))")
        end
    end
end


function main(system::String)
    Nlow = 16
    Naux = Nlow * 16
    l = 200
    thresh = 1e-3
    n_final = 10

    # filename = "../../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat" # personal
    filename = filename = "../Davidson_algorithm/m_pp_" * system * ".dat"
    A = load_matrix(system, filename)
    N = size(A, 1)

    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i, i] = 1.0
    end

    println("Starting Davidson solver")
    @time Σ, U = davidson_driver(A, V, Naux, l, thresh, n_final)

    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]

    # println("Full diagonalization (reference)")
    # @time Σexact, _ = eigen(A)

    # display("text/plain", (Σ - Σexact[1:l])')
end

main("He")
