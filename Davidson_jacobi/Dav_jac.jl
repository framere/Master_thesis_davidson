using LinearAlgebra
using IterativeSolvers

# ------------------------------------------------------------
# Helper function: project vector orthogonal to v
function project(v::AbstractVector, x::AbstractMatrix)
    return x - v * v' * x
end

# ------------------------------------------------------------
# Helper function: project matrix-vector product operator
function M_operator(A, v, λ)
    I_mat = I(size(A, 1))  # Identity matrix of the same size as A
    x = (A - λ * I_mat)
    y = project(v, x) 
    z = project(v, y)
    return z
end

# ------------------------------------------------------------
# Main function: Block Jacobi-Davidson
function block_jacobi_davidson(A::AbstractMatrix; nev=2, tol=1e-4, maxiter=100)
    n = size(A,1)
    V = zeros(n, nev)
    for i = 1:nev
        V[i,i] = 1.0
    end
    D = diag(A)

    converged = falses(nev) # convergence flags for each eigenpair

    λs = zeros(nev)
    
    iter = 0
    while !all(converged) && iter < maxiter
        iter += 1

        # orthogonalize guess orbitals (using QR decomposition)
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)    

        # construct and diagonalize Rayleigh matrix
        H = Hermitian(V'*(A*V))
        Σ, U = eigen(H, 1:nev)

        X = V*U # Ritz vecors
        R = X.*Σ' - A*X # residual vectors

        # Solve correction equations for each unconverged eigenpair
        S = zeros(n, nev)
        for i in 1:nev
            if converged[i]
                continue
            end
            v = V[:,i]
            λ = λs[i]
            r = R[i]
            
            # Define LinearOperator M
            Mop = M_operator(A, v, λ)
            
            # Preconditioner: inverse diagonal
            diagA = diag(A)
            precond_vec = diagA .- λ
            precond = 1 ./ precond_vec
            Pl = Matrix(Diagonal(precond))

            
            # Solve M s = -r using GMRES
            s, _ = gmres(Mop, -r; Pl=Pl, log=true, reltol=1e-8)
            S[:,i] = s
        end

        # Build new subspace
        W = hcat(V, S)
        # Orthonormalize W
        Q, _ = qr(W)
        Q = Matrix(Q)
        
        # Project A into subspace
        Ahat = Q' * A * Q
        
        # Solve small eigenproblem
        D, T = eigen(Ahat)
        
        # Rotate back
        V = Q * T
        
        # Update residuals and convergence
        for i in 1:nev
            r_norm = norm(A*V[:,i] - D[i]*V[:,i])
            if r_norm < tol
                converged[i] = true
            end
        end
        
        println("Iteration $iter: residual norms = ", [norm(A*V[:,i] - D[i]*V[:,i]) for i in 1:nev])
    end
    
    return D[1:nev], V[:,1:nev]
end

# ------------------------------------------------------------
# Example usage
n = 50
A = Hermitian(randn(n,n))
λs, Vs = block_jacobi_davidson(A; nev=2, tol=1e-8, maxiter=50)

println("Computed eigenvalues: ", λs)
