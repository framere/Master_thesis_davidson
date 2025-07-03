using LinearAlgebra
using Printf

include("functions_davidson.jl")

function orthogonalize(V::Matrix{T}, V_lock::Matrix{T}, weight::Float64=0.0) where T
    if isempty(V_lock)
        return qr(V).Q
    else
        # Soft deflation: project out locked vectors with reduced weight
        V .= V .- weight .* (V_lock * (V_lock' * V))
        Q, _ = qr(V)
        return Q
    end
end

function select_corrections_ORTHO(t::Matrix{T}, V::Matrix{T}, V_lock::Matrix{T}, 
                                thresh::Float64, eps::Float64, weight::Float64=0.0) where T
    # Apply soft deflation first
    if !isempty(V_lock)
        t .= t .- weight .* (V_lock * (V_lock' * t))
    end
    
    # Then orthogonalize against current basis
    t .= t .- V * (V' * t)
    
    # Select significant corrections
    norms = vec(mapslices(norm, t, dims=1))
    mask = norms .> thresh
    t_hat = t[:, mask]
    
    # Re-normalize
    for i in 1:size(t_hat, 2)
        nrm = norm(t_hat[:, i])
        if nrm > eps
            t_hat[:, i] ./= nrm
        end
    end
    
    return t_hat, size(t_hat, 2)
end

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
    soft_deflation_weight = 0.1  # Weight for soft deflation (0.1 means 10% of converged vectors remain)

    while nevf < l
        iter += 1
        remaining = l - nevf

        if remaining <= n_final
            println("-----Final convergence mode (soft deflation)-----")
            while true
                iter += 1

                # Orthogonalize with soft deflation
                V = orthogonalize(V, V_lock, soft_deflation_weight)

                Σ, X, R = rayleigh_ritz_projection(A, V, remaining)
                Rnorm = norm(R, 2)

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
                
                # Correction vectors with soft deflation awareness
                t = zero(similar(R))
                for i = 1:size(t, 2)
                    # Modified preconditioner that accounts for soft deflation
                    θ = Σ[i]
                    denom = θ .- D
                    # Apply soft deflation to the preconditioner
                    if !isempty(V_lock)
                        proj = V_lock * (V_lock' * R[:, i])
                        denom .+= soft_deflation_weight * abs.(proj) ./ (θ .- D .+ 1e-10)
                    end
                    t[:, i] = R[:, i] ./ denom
                end

                # Update guess basis
                if size(V, 2) <= Naux - remaining
                    V = hcat(V, t)
                else
                    # Keep some of the converged directions with soft weight
                    V = hcat(X, t)
                end

                if nevf >= l
                    println("Converged all eigenvalues.")
                    return Eigenvalues, Ritz_vecs
                end
            end 
        
        else
            # Standard block iteration with soft deflation
            V = orthogonalize(V, V_lock, soft_deflation_weight)

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

            # Correction vectors with soft deflation
            t = Matrix{T}(undef, size(A, 1), length(non_conv_indices))
            for (i, idx) in enumerate(non_conv_indices)
                θ = Σ_nc[i]
                denom = θ .- D
                # Soft deflation contribution
                if !isempty(V_lock)
                    proj = V_lock * (V_lock' * R_nc[:, i])
                    denom .+= soft_deflation_weight * abs.(proj) ./ (θ .- D .+ 1e-10)
                end
                t[:, i] = R_nc[:, i] ./ denom
            end

            # Orthogonalize corrections with soft deflation
            T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-10, soft_deflation_weight)

            # Update subspace
            if size(V, 2) + n_b_hat > Naux || !isempty(conv_indices) || n_b_hat == 0 
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
    l = 108
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

    println("Full diagonalization (reference)")
    @time Σexact, _ = eigen(A)

    display("text/plain", (Σ - Σexact[1:l])')
end

main("hBN")
