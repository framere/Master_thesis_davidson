using LinearAlgebra
using Printf

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    blocks::Int
)::Tuple{Vector{T},Matrix{T}} where T<:Number

    Nlow = size(V,2)
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)
    Ritz_vecs = []
    Eigenvalues = Float64[]
    Xconv = Matrix{T}(undef, size(A,1), 0)  # Empty orthonormal basis

    for block in 1:blocks
        println("Block ", block, " of ", blocks)
        iter = 0
        converged_in_block = 0

        while converged_in_block < Nlow && length(Eigenvalues) < Nlow * blocks
            iter += 1
            qr_decomp = qr(V)
            V = Matrix(qr_decomp.Q)

            H = Hermitian(V' * (A * V))
            Σ, U = eigen(H, 1:Nlow)

            X = V * U
            R = X .* Σ' .- A * X
            Rnorm = norm(R, 2)

            output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
            print(output)

            for i = 1:Nlow
                if norm(R[:, i]) < thresh
                    # Check if this eigenvalue is already found
                    is_duplicate = any(norm(Xconv' * X[:, i]) .> 0.99)
                    if is_duplicate
                        continue
                    end

                    push!(Ritz_vecs, X[:, i])
                    push!(Eigenvalues, Σ[i])
                    Eigenvalue_number = length(Eigenvalues)
                    println("converged eigenvalue ", Σ[i], " with residual norm ", norm(R[:, i]), " (eigenvalue number: ", Eigenvalue_number, ")")
                   
                    # Orthonormalize and add to Xconv
                    q = X[:, i]
                    if size(Xconv, 2) > 0
                        q -= Xconv * (Xconv' * q)
                    end
                    q /= norm(q)
                    Xconv = hcat(Xconv, q)
                    converged_in_block += 1
                end
            end

            if converged_in_block >= Nlow
                break
            end

            # Remove converged space from V
            for i in 1:size(V,2)
                V[:, i] -= Xconv * (Xconv' * V[:, i])
            end

            # Preconditioned residual
            t = zero(similar(R))
            for i = 1:size(t,2)
                C = 1.0 ./ (Σ[i] .- D)
                t[:, i] = C .* R[:, i]
            end

            # Project out converged space from t
            if size(Xconv, 2) > 0
                for i = 1:size(t,2)
                    t[:, i] -= Xconv * (Xconv' * t[:, i])
                end
            end

            # Augment V
            if size(V,2) <= Naux - Nlow
                V = hcat(V, t)
            else
                V = hcat(X, t)
            end
        end
    end

    return (Eigenvalues, hcat(Ritz_vecs...))
end


function main(system::String)
    # the two test systems He and hBN are hardcoded
    system = system
    
    Nlow = 16 # we are interested in the first Nlow eigenvalues
    Naux = Nlow * 16 # let our auxiliary space be larger (but not too large)

    if system == "He"
        N = 4488
    elseif system == "hBN"
        N = 6863        
    elseif system == "Si"
        N = 6201
    else
        println("Systen ", system, " unknown.")
        exit()
    end

    # read the matrix
    filename = "../Davidson_algorithm/m_pp_" * system * ".dat" #
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N*N)
    read!(file, A)
    close(file)
    A = reshape(A, N, N)
    A = -A # because we are interested in the largest eigenvalues
    A = Hermitian(A)

    # initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # perform Davidson algorithm
    println("Davidson")
    n_blocks = 3 # number of blocks to split the Davidson algorithm into
    @time Σ, U = davidson(A, V, Naux, 1e-5, n_blocks)
    # idx = sortperm(Σ)
    # Eigenvalues = Eigenvalues[idx] # they are not sorted! 
    # Ritz_vecs = Ritz_vecs[:,idx] # sort the converged eigenvectors

    
    # perform exact diagonalization as a reference
    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A) 

    display("text/plain", Σexact[1:n_blocks*Nlow]')
    display("text/plain", Σ')
    display("text/plain", (Σ[1:n_blocks*Nlow]-Σexact[1:n_blocks*Nlow])')
end

main("He")
