using LinearAlgebra
using Random

function Hamiltonian(N, L = 1.0)
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

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # diagonal part of A (for preconditioner)
    D = diag(A)
    # iterations
    iter = 0
    while true
        iter = iter + 1
        
        # orthogonalize guess orbitals (using QR decomposition)
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # construct and diagonalize Rayleigh matrix
        H = Hermitian(V'*(A*V))
        Σ, U = eigen(H, 1:Nlow)

        X = V*U # Ritz vecors
        R = X.*Σ' - A*X # residual vectors
        Rnorm = norm(R,2) # Frobenius norm
        
        # status output
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)
        
        if Rnorm < thresh
            println("converged!")
            return (Σ, X)
        end

        # update guess space using diagonal preconditioner 
        t = zero(similar(R)) 
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D) 
            t[:,i] = C .* R[:,i] # the new basis vectors
        end

        # update guess basis
        if size(V,2) <= Naux-Nlow
            V = hcat(V,t) # concatenate V and t
        else
            V = hcat(X,t) # concatenate X and t 
        end
    end
end


function main()
    N = 100  # Number of grid points
    L = 1.0  # Length of the domain
    H = Hamiltonian(N, L)

    # Initial guess vector (normalized)
    v1 = randn(N)
    v1 /= norm(v1)

    M = 16  # Number of eigenvalues to find
    iterations_count = 20
    thresh = 1e-6

    # Initial guess space
    V = hcat(v1, randn(N, M-1))

    # Run Davidson algorithm
    eigenvalues, eigenvectors = davidson(H, V, N + M, thresh)

    println("Eigenvalues: ", eigenvalues)
end

main()