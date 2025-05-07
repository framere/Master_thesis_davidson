using LinearAlgebra
using Printf
using Random

# function sparse_matrix(N::Int, factor::Int)
#     A = zeros(Float64, N, N)
#     for i in 1:N
#         for j in 1:N
#             if i == j
#                 A[i, j] = rand() * factor
#             else
#                 if rand() < 0.05 # 5% chance to be non-zero off-diagonal --> Problemmm
#                     A[i, j] = rand() / factor
#                 else
#                     # Keep the off-diagonal elements as zero
#                     A[i, j] = 0.0
#                 end
#             end
#         end
#     end
#     A =0.5 * (A + A')

#     return Hermitian(A)
# end


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
function main(factor::Int)
    N = 6863
    factor = factor
    println("Generating sparse matrix of size ", N, " with factor ", factor)
    A = sparse_matrix(N, factor)

    Nlow = 16 # we are interested in the first Nlow eigenvalues
    Naux = Nlow * 16 # let our auxiliary space be larger (but not too large)

    # Perform Davidson algorithm
    V = zeros(N, Nlow)
    for i in 1:Nlow
        V[i, i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, 1e-3, factor)

    println("Lanczos")
    @time Σ_lanczos, U_lanczos = lanczos(A, V, Naux, 1e-3, factor)


end

# a simple implementation of the block Davidson method for a Hermitian matrix A
function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    factor:: Int # default system is hBN
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # diagonal part of A (for preconditioner)
    D = diag(A)
    logfile = open("Plots_test_matrix/davidson_test_factor_$factor.txt", "w") 
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
        @printf(logfile, "%d %.6e\n", iter, Rnorm)
        print(output)
        
        if Rnorm < thresh
            println("converged!")
            close(logfile)
            return (Σ, X)
        end

        # update guess space using diagonal preconditioner 
        t = zero(similar(R)) 
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D) # diagonal preconditioner
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


function lanczos(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    factor::Int # default system is hBN    
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # diagonal part of A (for preconditioner)
    D = diag(A)
    logfile = open("Plots_test_matrix/Lanczos_log_factor_$factor.txt", "w")

    # iterations
    iter = 0
    while true
        iter = iter + 1
        
        # orthogonalize guess orbitals (using QR decomposition)
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # construct and diagonalize Rayleigh matrix
        H = Hermitian(V'*(A*V))
        Σ, U = eigen(H, 1:Nlow) #Eigenvalues (vector) and Eigenvectors (matrix)

        X = V*U # Ritz vectors
        R = X.*Σ' - A*X # residual vectors (matrix)
        Rnorm = norm(R,2) # Frobenius norm (same as Euclidian but for matrices)
		
        # status output
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        @printf(logfile, "%d %.6e\n", iter, Rnorm)
        print(output)
        
        if Rnorm < thresh
            println("converged!")
            close(logfile)
            return (Σ, X)
        end

        # update guess space using diagonal preconditioner 
        t = zero(similar(R)) 
        for i = 1:size(t,2)
#            C = 1.0 ./ Tridiagonal(.- dl )
            t[:,i] = R[:,i] # the new basis vectors
        end

        # update guess basis
        if size(V,2) <= Naux-Nlow
            V = hcat(V,t) # concatenate V and t
        else
            V = hcat(X,t) # concatenate X and t 
        end
    end
end


main(300)

# for i in collect(10:10:100)
#     main(i)
# end
