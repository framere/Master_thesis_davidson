using LinearAlgebra
using IterativeSolvers
using LinearOperators

function davidson_jacobi(A; m=20, tol=1e-8, max_iters=100, Nlow=3)
    n = size(A, 1)
    V = Matrix(qr(randn(n, Nlow)).Q)  # Start with 3 orthonormal vectors

    iter_count = 0
    converged = false

    while !converged && iter_count < max_iters
        iter_count += 1

        # Orthonormalize V again (just in case)
        V = Matrix(qr(V).Q)

        # Projected matrix
        H = Hermitian(V' * (A * V))
        eigvals, eigvecs = eigen(H)

        # Get the Nlow smallest eigenpairs
        idxs = sortperm(eigvals)[1:Nlow]
        θs = eigvals[idxs]
        Ss = eigvecs[:, idxs]
        Xs = V * Ss  # Ritz vectors (each column is one Ritz vector)

        Rs = A * Xs - Xs .* θs'  # Residuals for each eigenpair
        Rnorm = norm(Rs)  # Frobenius norm
        println("iter=$iter_count  Rnorm=$Rnorm  dim(V)=$(size(V, 2))")

        if Rnorm < tol
            converged = true
            break
        end

        # Solve correction equation for each residual vector
        for i in 1:Nlow
            u = Xs[:, i]
            r = Rs[:, i]
            θ = θs[i]

            # Define projection operator P = I - uuᵗ
            function P(x)
                return x - u * (u' * x)
            end

            # Linear operator M = P*(A - θI)*P
            function M_mul!(y, x)
                tmp = P(x)
                tmp2 = (A - θ * I) * tmp
                y .= P(tmp2)
            end

            M = LinearOperator(Float64, n, n, false, false, M_mul!)

            # Solve correction equation approximately
            t_i, _ = bicgstabl(M, -r; reltol=1e-4, max_mv_products=1000)

            # Expand basis only with meaningful corrections
            if isa(t_i, AbstractVector) && norm(t_i) > 1e-12
                t_i ./= norm(t_i)
                V = hcat(V, t_i)
            end
        end
    end

    println("Converged in $iter_count iterations.")
    return θs, Xs
end

# Example usage:
n = 50
A = randn(n, n)
A = 0.5 * (A + A')  # Hermitian
θs, Xs = davidson_jacobi(A)
println("Eigenvalues found: ", θs)
