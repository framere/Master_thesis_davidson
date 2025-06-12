using LinearAlgebra
using Printf

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    system::String  # default system
)::Tuple{Vector{T},Matrix{T}} where T<:Number

    Nlow = size(V,2) # number of desired eigenvalues
    if Naux < Nlow
        error("Auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)                # diagonal part for preconditioning
    logfile = open("davidson_log_$system.txt", "w")

    # Locked (converged) eigenvalues and eigenvectors
    Σconv = T[]                               # converged eigenvalues
    Xconv = Matrix{T}(undef, size(A,1), 0)     # converged eigenvectors

    iter = 0
    while true
        iter += 1

        # Project out converged vectors from V
        if size(Xconv, 2) > 0
            V = V - Xconv * (Xconv' * V)
        end
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q) # Orthonormalize

        # Build Rayleigh-Ritz matrix
        H = Hermitian(V' * (A * V))
        Σtrial, U = eigen(H, 1:Nlow - length(Σconv)) # unconverged part only
        X = V * U                                   # Ritz vectors

        # Residuals
        R = X .* Σtrial' .- A * X
        norms = [norm(R[:,i]) for i in 1:size(R,2)]
        Rnorm = norm(R,2)

        # Logging and output
        @printf(logfile, "%d %.6e\n", iter, Rnorm)
        println(@sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d", iter, Rnorm, size(V,2)))

        # Lock converged eigenpairs
        keep = trues(length(Σtrial))
        for i in 1:length(Σtrial)
            if norms[i] < thresh
                push!(Σconv, Σtrial[i])
                Xconv = hcat(Xconv, X[:,i])
                keep[i] = false
            end
        end

        # Convergence check
        if length(Σconv) >= Nlow
            println("Converged!")
            close(logfile)
            return (Σconv, Xconv)
        end

        # Prepare expansion vectors
        R = R[:, keep]
        Σtrial = Σtrial[keep]
        t = Matrix{T}(undef, size(A,1), length(Σtrial))
        for i in 1:length(Σtrial)
            C = 1.0 ./ (Σtrial[i] .- D)
            t[:,i] = C .* R[:,i]
        end

        # Expand subspace
        if size(V,2) + size(t,2) <= Naux
            V = hcat(V, t)
        else
            # Restart: use locked + most recent Ritz + new expansion
            V = hcat(X, t)
        end
    end
end


function define_matrix(system::String)
    # Define a sample matrix for testing
    
    Nlow = 120   # we are interested in the first Nlow eigenvalues
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
    filename = "../Davidson_algorithm/m_pp_" * system * ".dat" #institute
    # filename = "../../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat" # personal
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
    @time Σ, U = davidson(A, V, Naux, 1e-5, system)
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

main("hBN")  # Change to "hBN" or "Si" as needed