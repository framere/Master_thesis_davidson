using LinearAlgebra
using Printf

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Nauxiliary::Integer,
    thresh::Float64,
    target_nev::Int,
    deflation_eps::Float64
)::Tuple{Vector{T},Matrix{T}} where T<:Number

    Nlow = size(V,2)
    Ritz_vecs = []
    Eigenvalues = Float64[]
    Xconv = Matrix{T}(undef, size(A,1), 0)  # Empty orthonormal basis

    block = 0
    while length(Eigenvalues) < target_nev
        block += 1
        println("Block ", block)
        iter = 0
        D = diag(A)  # Diagonal part of A (for preconditioner)
        Naux = copy(Nauxiliary) # Ensure Naux is mutable
        sizeV = size(V, 2) 
        println("Initial size of V for block ", block, " is ", sizeV)
        println("Number of eigenvalues to find in this block: ", Nlow)
        println("Number of auxiliary vectors: ", Naux)

        while true
            iter += 1
            qr_decomp = qr(V)
            V = Matrix(qr_decomp.Q)

            if size(Xconv, 2) > 0
                V = V - Xconv * (Xconv' * V)
                V = Matrix(qr(V).Q)
            end

            H = Hermitian(V' * (A * V))
            Σ, U = eigen(H, 1:Nlow)

            X = V * U
            R = X .* Σ' .- A * X

            Rnorm = norm(R, 2)

            output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
            print(output)

            if Rnorm < thresh
                if size(Xconv, 2) > 0
                    proj_norm = norm(Xconv' * X, 2)
                else
                    proj_norm = 0.0
                end

                if proj_norm < 1e-1
                    println("converged block ", block, " with Rnorm ", Rnorm)
                    for i = 1:Nlow
                        if abs.(Σ[i] - Σ[end]) .> deflation_eps* abs(Σ[end])
                            push!(Ritz_vecs, X[:, i])
                            push!(Eigenvalues, Σ[i])
                            @printf("converged eigenvalue %.3f with residual norm %.2e\n", Σ[i], norm(R[:, i]))
                            
                            q = X[:, i]
                            if size(Xconv, 2) > 0
                                q -= Xconv * (Xconv' * q)
                            end
                            q /= norm(q)
                            Xconv = hcat(Xconv, q)
                        else
                            # Deflation: eigenvalue is too close to converged eigenvalues
                            @printf("Deflation eigenvalue %.3f: cutting through degenerate eigenvalues\n", Σ[i])
                            continue
                        end
                    end
                # remove from X conv the vectors, whose eigenvalues are too close to the converged eigenvalues
                end
                break # exit the loop after finding Nlow eigenvalues
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

    end

    return (Eigenvalues, hcat(Ritz_vecs...))
end



function define_matrix(system::String)
    # Define a sample matrix for testing
    
    Nlow = 25 # we are interested in the first Nlow eigenvalues
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
    # filename = "../Davidson_algorithm/m_pp_" * system * ".dat" #institute
    filename = "../../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat" # personal
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N*N)
    read!(file, A)
    close(file)
    A = reshape(A, N, N)
    A = -A # because we are interested in the largest eigenvalues
    A = Hermitian(A)
    return A, N, Nlow, Naux
end


function main(system::String)
    A, N, Nlow, Naux = define_matrix(system)

    # initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # perform Davidson algorithm
    println("Davidson")
    target_nev = 80 # number of blocks to split the Davidson algorithm into
    @time Σ, U = davidson(A, V, Naux, 1e-5, target_nev, 1e-2)
    idx = sortperm(Σ)
    Σ = Σ[idx] # they are not sorted! 
    # Ritz_vecs = Ritz_vecs[:,idx] # sort the converged eigenvectors

    
    # # perform exact diagonalization as a reference
    # println("Full diagonalization")
    # @time Σexact, Uexact = eigen(A) 

    # display("text/plain", Σexact[1:length(Σ)])
    # display("text/plain", Σ')
    # display("text/plain", (Σ-Σexact[1:length(Σ)])')
end

main("hBN")
