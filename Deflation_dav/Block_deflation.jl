using LinearAlgebra
using Printf

# generate sparse matrix
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

Matrix = sparse_matrix(1000, 100)
display(Matrix)


function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    system::String, # default system is hBN
    blocks::Int # number of blocks for block Davidson
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # diagonal part of A (for preconditioner)
    D = diag(A)
    Ritz_vecs = [] # Ritz vectors
    Eigenvalues = Float64[] # Ritz eigenvalues

    for block in 1:blocks
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
                for i = 1:size(X,2)
                    push!(Ritz_vecs, X[:,i])
                    push!(Eigenvalues, Σ[i])
                    println("converged eigenvalue ", Σ[i], " with norm ", norms[i])
                end
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

        Xconv = hcat(Ritz_vecs...) # converged eigenvectors
        for j in 1:size(V,2)
            V[:, j] -= Xconv * (Xconv' * V[:, j])  # project out component in span(Xconv)
        end

        return (Eigenvalues, Ritz_vecs)
    end
end