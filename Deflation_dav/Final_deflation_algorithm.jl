using LinearAlgebra
using Printf

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    blocks::Int
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    Nlow = size(V, 2)
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)
    Ritz_vecs = []
    Eigenvalues = Float64[]
    Xconv = Matrix{T}(undef, size(A, 1), 0)  # Converged subspace

    iter = 0
    while length(Eigenvalues) < Nlow 
        iter += 1
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        H = Hermitian(V' * (A * V))
        Σ, U = eigen(H, 1:Nlow)
        X = V * U
        R = A * X .- X .* Σ'
        Rnorm = norm(R,2)

        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V, 2))
        print(output)

        for i = 1:Nlow
            rnorm = norm(R[:, i])
            if rnorm < thresh
                # Check if vector is new (not already in Xconv)
                if size(Xconv, 2) > 0
                    proj_norm = norm(Xconv' * X[:, i])
                else
                    proj_norm = 0.0
                end

                if proj_norm < 1e-1  # check that the scalar product is small enough
                    println("converged eigenvalue ", Σ[i], " with residual norm ", rnorm, "// Eigenvalue number: ", length(Eigenvalues) + 1)
                    push!(Ritz_vecs, X[:, i])
                    push!(Eigenvalues, Σ[i])

                    # Orthonormalize and add to Xconv
                    q = X[:, i]
                    if size(Xconv, 2) > 0
                        q -= Xconv * (Xconv' * q)
                    end
                    q /= norm(q)
                    Xconv = hcat(Xconv, q)
                end
            end
        end

        # Form correction vectors
        t = zero(similar(R)) 
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D) 
            t[:,i] = C .* R[:,i] # the new basis vectors
        end

        # Adaptive deflation: only project if overlap is significant
        # if size(Xconv, 2) > 0
        #     for i = 1:size(t, 2)
        #         overlap = Xconv' * t[:, i]
        #         if norm(overlap) > 1e-3
        #             t[:, i] -= Xconv * overlap
        #         end
        #     end
        # end

        if size(V, 2) <= Naux - Nlow
            V = hcat(V, t)
        else
            V = hcat(X, t)
        end
    end

    return (Eigenvalues, hcat(Ritz_vecs...))
end



function define_matrix(system::String)
    # Define a sample matrix for testing
    
    Nlow = 100   # we are interested in the first Nlow eigenvalues
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
    n_blocks = 1 # number of blocks to split the Davidson algorithm into
    @time Σ, U = davidson(A, V, Naux, 1e-5, n_blocks)
    # sorted_indices = sortperm(Σ)
    # Eigenvalues = Σ[sorted_indices]  # Sort eigenvalues
    # Ritz_vecs = U[:, sorted_indices]  # Sort eigenvectors
    # # perform exact diagonalization as a reference
    # println("Full diagonalization")
    # @time Σexact, Uexact = eigen(A) 

    # display("text/plain", Σexact[1:n_blocks*Nlow]')
    # display("text/plain", Σ')
    # display("text/plain", (Eigenvalues[1:n_blocks*Nlow] -Σexact[1:n_blocks*Nlow])')
end

systems = ["He", "hBN", "Si"]

for system in systems
    println("system: ", system)
    main(system)
end


# main("hBN")