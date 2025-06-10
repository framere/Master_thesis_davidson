using LinearAlgebra
using Printf

# generate symmetric matrix
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

    for block in 1:blocks
        iter = 0
        while true
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

            converged = [norm(R[:, i]) < thresh for i = 1:Nlow]
            if all(converged)
                for i = 1:Nlow
                    push!(Ritz_vecs, X[:, i])
                    push!(Eigenvalues, Σ[i])
                    println("converged eigenvalue ", Σ[i], " with residual norm ", norm(R[:, i]))
                end
                break
            end

            t = zero(similar(R))
            for i = 1:size(t,2)
                C = 1.0 ./ (Σ[i] .- D)
                t[:, i] = C .* R[:, i]
            end

            if size(V,2) <= Naux - Nlow
                V = hcat(V, t)
            else
                V = hcat(X, t)
            end
        end

        Xconv = qr(hcat(Ritz_vecs...)).Q
        for j in 1:size(V,2)
            V[:, j] -= Xconv * (Xconv' * V[:, j])
        end
    end

    return (Eigenvalues, hcat(Ritz_vecs...))
end

# -----------------------------------------------------------------------------------
# function main()
#     A = sparse_matrix(10000, 10)  # Example sparse matrix

#     Nlow = 16
#     Naux = Nlow * 16
#     N = size(A, 1)

#     V = zeros(N, Nlow)
#     for i = 1:Nlow
#         V[i, i] = 1.0
#     end

#     println("Davidson")
#     @time Σ, U = davidson(A, V, Naux, 1e-5, 2)
# end


# main()


# -----------------------------------------------------------------------------------

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
    n_blocks = 2 # number of blocks to split the Davidson algorithm into
    @time Σ, U = davidson(A, V, Naux, 1e-5, n_blocks)
    # idx = sortperm(Σ)
    # Eigenvalues = Eigenvalues[idx] # they are not sorted! 
    # Ritz_vecs = Ritz_vecs[:,idx] # sort the converged eigenvectors

    
    # perform exact diagonalization as a reference
    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A) 

    display("text/plain", Σexact[1:n_blocks*Nlow]')
    display("text/plain", Σ')
    display("text/plain", (Σ-Σexact[1:n_blocks*Nlow])')
    return Σ, U
end

main("He")
