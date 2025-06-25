using LinearAlgebra
using IterativeSolvers
using LinearOperators


function jacobi_filter(A::AbstractMatrix{Float64}, V::Matrix{Float64}, n::Int; 
                       tol=1e-6, N_block=4, max_iters=100)
    N = size(A, 1)
    λ = zeros(n)
    k = 0  # index of the lowest unconverged eigenvector
    V = qr(V).Q[:, 1:n]  # Orthonormalize initial V

    for iter = 1:max_iters
        if k >= n
            println("Converged after $iter iterations.")
            break
        end

        # Define block indices
        block_start = k + 1
        block_end = min(k + N_block, n)
        block_inds = block_start:block_end
        block_size = length(block_inds)

        println("\n▶ Iteration $iter — Updating eigenvectors $block_start to $block_end")

        # Compute Rayleigh quotients and residuals
        r = Matrix{Float64}(undef, N, block_size)
        λ_block = zeros(block_size)
        for (i, j) in enumerate(block_inds)
            vi = V[:, j]
            λi = dot(vi, A * vi)
            λ_block[i] = λi
            r[:, i] = A * vi - λi * vi
            println("   λ[$j] ≈ $(round(λi, digits=6)) |residual| = $(round(norm(r[:, i]), digits=2))")
        end

        # Solve correction equations with MINRES
        S = zeros(N, block_size)


        for (i, j) in enumerate(block_inds)
            vi = V[:, j]
            λi = λ_block[i]
            P = I - vi * vi'

            function mv(s)
                return P * (A * s - λi * s)
            end

            Aop = LinearOperator(Float64, N, N, mv)

            println("   Solving correction equation for vector $j using MINRES...")
            s, _ = minres(Aop, r[:, i], reltol=1e-6, maxiter=100)

            S[:, i] = s
        end




        # Build subspace W and orthonormalize
        W = hcat(V[:, k+1:block_end], S[:, 1:block_size])
        for j in 1:k
            W = W - V[:, j] * (V[:, j]' * W)  # Orthogonalize against converged
        end
        W = qr(W).Q  # Orthonormalize

        # Project A into subspace
        A_sub = W' * A * W

        # Diagonalize subspace matrix
        eigvals, eigvecs = eigen(A_sub)

        # Rotate eigenvectors in subspace
        V[:, k+1:k+block_size] = W * eigvecs[:, 1:block_size]
        λ[k+1:k+block_size] = eigvals[1:block_size]

        # Check convergence
        new_k = k
        for j in k+1:k+block_size
            rj = A * V[:, j] - λ[j] * V[:, j]
            if norm(rj) > tol
                new_k = j
            end
        end

        k = new_k + 1
        println("   Diagonalizing subspace of size $(size(W, 2))")
        
    end

    return V[:, 1:n], λ[1:n]
end


A = randn(100, 100)
A = Hermitian(A + A')

V0 = randn(100, 10)  # Initial guess for 10 eigenvectors

V, λ = jacobi_filter(A, V0, 10)
