using LinearAlgebra
using IterativeSolvers
using Printf

function jacobi_davidson_block(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64;
    max_iter::Int = 100,
)::Tuple{Vector{T},Matrix{T}} where T<:Number

    n = size(A,1)
    Nlow = size(V, 2)
    if Naux < Nlow
        error("Auxiliary basis size must be >= number of eigenvalues requested.")
    end

    D = diag(A)  # Diagonal of A (for preconditioner)
    iter = 0

    while iter < max_iter
        iter += 1

        # QR orthonormalization of current basis V
        V = Matrix(qr(V).Q)

        # Rayleigh-Ritz projection
        H = Hermitian(V' * A * V)
        λs, U = eigen(H, 1:Nlow)
        X = V * U                   # Ritz vectors
        R = A * X .- X .* λs'      # Residuals
        Rnorm = norm(R)            # Frobenius norm

        println(@sprintf("iter=%6d  Rnorm=%11.3e  dim(V)=%6d", iter, Rnorm, size(V,2)))
        if Rnorm < thresh
            println("Converged!")
            return λs, X
        end

        # Block Jacobi-Davidson correction
        Tblock = Matrix{T}(undef, n, Nlow)
        for i in 1:Nlow
            ri = R[:,i]
            vi = X[:,i]
            λi = λs[i]

            # Projector: P = I - vi*vi'
            Pi = I - vi * vi'

            # Approximate (I - vi*vi') (A - λi I)^(-1) (I - vi*vi') * ri
            M_diag_inv = 1.0 ./ (D .- λi)      # Diagonal preconditioner
            zi = M_diag_inv .* (Pi * ri)      # Apply preconditioner to projected residual
            si = Pi * zi                      # Project again to stay orthogonal to vi

            Tblock[:,i] = si
        end

        # Expand search space with QR orthonormalization
        if size(V, 2) <= Naux - Nlow
            V = hcat(V, Tblock)
        else
            V = hcat(X, Tblock)
        end
    end

    error("Jacobi-Davidson did not converge within $max_iter iterations.")
end


function main(system::String)
    if system == "He"
        N = 4488
    elseif system == "hBN"
        N = 6863        
    elseif system == "Si"
        N = 6201
    else
        error("Unknown system: $system")
    end

    
    Nlow = 16 # we are interested in the first Nlow eigenvalues
    Naux = Nlow * 16 # let our auxiliary space be larger (but not too large)
    # read the matrix
    # filename = "../Davidson_algorithm/m_pp_" * system * ".dat"
    filename = "../../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat" # personal
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)
    A = reshape(A, N, N)
    A = Hermitian(A)

    # initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i,i] = 1.0
    end

    ## initial guess vectors (stochastic guess)
    #V = rand(N,Nlow) .- 0.5

    # perform Davidson algorithm
    println("Davidson")
    @time λs, X = jacobi_davidson_block(A, V, Naux, 1e-3)

    # perform exact diagonalization as a reference
    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A) 

    #display("text/plain", Σexact[1:Nlow]')
    #display("text/plain", Σ')
    display("text/plain", (λs-Σexact[1:Nlow])')
end

main("Si")