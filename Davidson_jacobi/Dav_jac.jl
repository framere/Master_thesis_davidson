using LinearAlgebra

# Rayleigh quotient λ = v'A*v
rayleigh_quotient(A, v) = dot(v, A * v)

# Residual r = A*v - λ*v
compute_residual(A, v, λ) = A * v .- λ * v

# Orthonormalize a matrix (columns)
function orthonormalize(V::AbstractMatrix)
    Q = copy(V)
    for i in 1:size(Q, 2)
        for j in 1:i-1
            Q[:, i] .-= dot(Q[:, j], Q[:, i]) * Q[:, j]
        end
        Q[:, i] ./= norm(Q[:, i])
    end
    return Q
end

# Block Jacobi–Davidson Davidson method with diagonal preconditioner
function block_jacobi_davidson(A::Hermitian{Float64, Matrix{Float64}}, n_eig::Int;
                               Nblock::Int = 2, max_iter::Int = 50, tol::Float64 = 1e-8)

    n = size(A, 1)
    D = diag(A)                        # Diagonal for preconditioner
    V = orthonormalize(randn(n, n_eig))  # Initial guess
    converged = falses(n_eig)
    λ = zeros(n_eig)
    residuals = [zeros(n) for _ in 1:n_eig]

    k = 1
    iter = 0

    while k ≤ n_eig && iter < max_iter
        iter += 1
        block_inds = k:min(k+Nblock-1, n_eig)

        # Step 1: Compute Rayleigh quotients and residuals
        for (j, idx) in enumerate(block_inds)
            vi = V[:, idx]
            λ[idx] = rayleigh_quotient(A, vi)
            residuals[idx] = compute_residual(A, vi, λ[idx])
        end

        # Step 2: Diagonal preconditioning for correction
        S = zeros(n, length(block_inds))
        for (j, idx) in enumerate(block_inds)
            vi = V[:, idx]
            ri = residuals[idx]
            λi = λ[idx]

            Pi = I - vi * vi'                         # Projector
            zi = (1.0 ./ (D .- λi)) .* (Pi * ri)      # Diagonal preconditioning
            si = Pi * zi                              # Final projection
            S[:, j] = si
        end

        # Step 3: Expand subspace and orthonormalize
        W = hcat(V[:, block_inds], S)
        W = orthonormalize(W)

        # Step 4: Rayleigh–Ritz in expanded subspace
        Atilde = W' * A * W
        evals, Q = eigen(Hermitian(Atilde))
        Vnew = W * Q[:, 1:length(block_inds)]

        # Step 5: Update eigenpairs
        for (j, idx) in enumerate(block_inds)
            V[:, idx] = Vnew[:, j]
            λ[idx] = evals[j]
            residuals[idx] = compute_residual(A, V[:, idx], λ[idx])
            converged[idx] = norm(residuals[idx]) < tol
        end

        # Step 6: Next unconverged
        next_unconverged = findfirst(!, converged)
        if isnothing(next_unconverged)
            break
        else
            k = next_unconverged
        end
    end

    return λ[1:n_eig], V[:, 1:n_eig]
end


function sparse_matrix(N::Int, factor::Int)
    A = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i, j] = rand() 
            else
                A[i, j] = rand() / factor
            end
        end
    end
    A = 0.5 * (A + A')

    return Hermitian(A)
end

A = sparse_matrix(1000, 300)
N_ev = 40
@time λ, V = block_jacobi_davidson(A, N_ev, Nblock=10)
println("Computed eigenvalues: ", λ)

# compare with exact results
exact_eigenvalues, exact_eigenvectors = eigen(A)

println("Exact eigenvalues: ", exact_eigenvalues[1:N_ev])
println("Difference: ", λ - exact_eigenvalues[1:N_ev])