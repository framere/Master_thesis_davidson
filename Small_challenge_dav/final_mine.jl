using LinearAlgebra 
using Printf

function Hamiltonian(N, L = 1.0)
    """
    Constructs the Hamiltonian matrix for a 1D system with periodic boundary conditions.
    The Hamiltonian includes a kinetic term and a quartic potential term.
    N: Number of grid points.
    L: Length of the system (default is 1.0).
    """
    H = zeros(Float64, N, N)
    dx_squared = L^2 / N^2
    for i in 1:N
        right = mod(i, N) + 1
        left = mod(i - 2, N) + 1
        H[i, i] = 1.0 / dx_squared
        H[i, right] = -0.5 / dx_squared
        H[i, left] = -0.5 / dx_squared
    end

    # Quartic term
    for i in 1:N
        argument = (i-1) - N/2
        H[i, i] += 1/24* argument^4 * dx_squared^2
    end
    return H
end

H = Hermitian(Hamiltonian(100))

println("Hamiltonian matrix constructed.")

function davidson(H::AbstractMatrix{Float64}, v_1::Vector{Float64}, M::Int; threshold=1e-1)
    """
    Davidson algorithm for finding the M lowest eigenvalues and eigenvectors 
    of the hermitian matrix H. We have an initial guess v_1 for the first eigenvector.
    There's a maximum number of iterations specified by `iterations`.
    """
    
    N = size(H, 1) # dimension of the Hamiltonian matrix
    D = diag(H)
    V = hcat(v_1) # matrix to store the eigenvectors
    lambda = [dot(v_1, H * v_1)] # initial Rayleigh quotient eigenvalue
    iter=1
    while true
        n = size(V, 2) # number of eigenvectors in V
        W = zeros(Float64, N, 2 * n) # matrix to store the residuals

        residuums = zeros(Float64, N, n) # initialize residual vector
        for k in 1:n
            W[:, k] = V[:, k]
            residuum = H * V[:, k] - lambda[k] * V[:, k] # residual vector
            delta_v = similar(residuum) # initialize delta_v
            delta_v .= -residuum ./ (D .- lambda[k]) # element-wise division
            W[:, n+k] = delta_v # store the correction vector
            residuums[:, k] = residuum
        end
        
        if norm(residuums, 2) < threshold
            @printf("Convergence achieved with residual norm: %.6f after iteration %d\n", norm(residuums, 2), iter)
            break # stop if the residual is below the threshold
        end

        Q, _ = qr(W) # QR decomposition

        U = Matrix(Q[:, 1:2*n]) # keep only the first 2*n columns

        J = U' * H * U # projected Hamiltonian
        J = Hermitian(J) # ensure J is hermitian
        m = min(M, 2*n)

        e, v = eigen(J) # compute eigenvalues and eigenvectors of J
        e = sort(e) # sort eigenvalues
        lambda = e[1:m] # take the first M eigenvalues
        v = v[:, 1:m] # take the first M eigenvectors
        
        v = U * v # back-transform eigenvectors

        V = hcat(v[:, 1:m]) # keep only the first M eigenvectors
        iter += 1
    end
    return V, lambda
end

H = Hermitian(Hamiltonian(100))

function main(H)
    N = size(H, 1) # number of grid points

    v1 = rand(Float64, N) # initial guess for the first eigenvector
    v1 /= norm(v1) # normalize the vector

    M = 10 # number of eigenvalues to find
    iterations = 100 # maximum number of iterations

    V, e = davidson(H, v1, M)
    true_eigenvalues, true_eigenvectors = eigen(H)

    # Print the eigenvalues
    println("Approxiation   True_Eigenvalues")
    for i in 1:M
        @printf("%d: %.6f       %.6f\n", i, e[i], true_eigenvalues[i])
    end
end

randMatrix = rand(Float64, 100, 100)
randMatrix = Hermitian(randMatrix)

main(H)