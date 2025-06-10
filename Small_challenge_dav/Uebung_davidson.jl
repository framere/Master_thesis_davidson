using LinearAlgebra
using Printf

const L = 10.0
matrix_vector_products_count = 0

function davidson(H::Matrix{Float64}, v1::Vector{Float64}, M::Int; iterations_count=20)
    N = size(H, 1)
    D = diag(H)

    # start guess vectors matrix
    V = zeros(Float64, N, 1)
    V[:, 1] = v1
    # initial Rayleigh quotient eigenvalues
    lambda = [dot(v1, H * v1)]
    
    open("psi.dat", "w") do psi_file
        for iteration in 1:iterations_count
            n = size(V, 2) # current subspace size (number of columns/vectors)
            W = zeros(Float64, N, 2 * n) # workspace for new vectors
            
            # for loop to find new vectors (loop over number of columns in V)
            for i in 1:n
                W[:, i] = V[:, i]
                global matrix_vector_products_count += 1
                residuum = H * V[:, i] - lambda[i] * V[:, i]
                delta_v = -residuum ./ (D .- lambda[i])  # Element-wise division
                W[:, n + i] = delta_v
            end

            
            # Orthogonalize columns of W using QR factorization
            Q, _ = qr(W)
            W = Matrix(Q[:, 1:(2*n)])  # Ensure size N x 2n
            
            # Project H onto subspace spanned by W
            h = zeros(Float64, 2*n, 2*n)
            for j in 1:2*n
                global matrix_vector_products_count += 1
                Hb = H * W[:, j] # Ritz vector
                for i in 1:j
                    h[i, j] = dot(W[:, i], Hb)
                    h[j, i] = h[i, j]
                end
            end
            
            # Diagonalize projected Hamiltonian
            eig = eigen(Symmetric(h))
            m = min(M, 2*n) # minimal of M and 2n
            
            # Ritz values and vectors for lowest m eigenvalues
            lambda = eig.values[1:m]
            V = W * eig.vectors[:, 1:m]
            
            # Write wavefunctions of ground and first excited state to file
            for k in 1:N
                x = (k-1) * L / N
                psi0 = V[k, 1]
                psi1 = m > 1 ? V[k, 2] : 0.0
                @printf(psi_file, "%.8e %.8e %.8e\n", x, psi0, psi1)
            end
            write(psi_file, "\n\n")
        end
    end
    
    return lambda, V
end

function main()
    N = 100
    H = zeros(Float64, N, N)
    DeltaX_squared = (L / N)^2
    
    # Setup Hamiltonian matrix with 3-point stencil and anharmonic potential
    for k in 1:N
        H[k, k] = 1.0 / DeltaX_squared
        
        right = mod(k, N) + 1
        left = mod(k - 2, N) + 1
        
        H[k, right] = -0.5 / DeltaX_squared
        H[k, left] = -0.5 / DeltaX_squared
        
        x = (k - 1) * L / N - L / 2
        H[k, k] += x^4 / 24
    end
    
    # Initial trial guess vector breaking symmetries
    v1 = zeros(Float64, N)
    for k in 1:N
        v1[k] = 1 + (k - 1) * L / N
    end
    v1 /= norm(v1)

    M = 16  # Subspace size for Davidson method
    
    # Run Davidson method with subspace size 16 and 100 iterations
    lambda, V = davidson(H, v1, M, iterations_count=2000)
    
    # Calculate exact eigenvalues with Julia built-in solver for comparison
    eigenvalues = eigen(Symmetric(H)).values
    eigenvectors = eigen(Symmetric(H)).vectors

   open("exact.dat", "w") do eigen_vectors_file
        n = size(eigenvectors, 1)  # number of rows (entries) in each eigenvector
        for i in 1:n
            @printf(eigen_vectors_file, "%e %e\n", eigenvectors[i, 1], eigenvectors[i, 2])
        end
    end


    println(" Ritz values   Eigenvalues")
    for i in 1:length(lambda)
        @printf("%e %e\n", lambda[i], eigenvalues[i])
    end
    
    global matrix_vector_products_count
    println("matrix-vector products: $matrix_vector_products_count")
    println("psi.dat contains plots")
end

main()
