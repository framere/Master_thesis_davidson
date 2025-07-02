using LinearAlgebra
using Printf

function load_matrix(system::String)
    if system == "He"
        N = 4488
    elseif system == "hBN"
        N = 6863        
    elseif system == "Si"
        N = 6201
    else
        error("Unknown system: $system")
    end

    filename = "../Davidson_algorithm/m_pp_" * system * ".dat"
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A  # for largest eigenvalues of original matrix
    return Hermitian(A)
end

function main(system::String)
    system = system
    Nlow = 200
    Naux = Nlow * 16

    A = load_matrix(system)
    N = size(A, 1)

    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i,i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, 1e-3, system)

    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A) 

    display("text/plain", Σ')
    display("text/plain", (Σ - Σexact[1:Nlow])')
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    system::String
)::Tuple{Vector{T},Matrix{T}} where T<:Number

    Nlow = size(V, 2)
    if Naux < Nlow
        error("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)
    converged = falses(Nlow)
    V_lock = Matrix{T}(undef, size(V, 1), 0)  # locked vectors

    iter = 0
    while true
        iter += 1

        # Orthonormalize
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # Rayleigh-Ritz
        H = Hermitian(V' * (A * V))
        Σ, U = eigen(H, 1:Nlow)
        X = V * U
        R = X .* Σ' .- A * X

        # Residual deflation: project out locked subspace
        if size(V_lock, 2) > 0
            R .-= V_lock * (V_lock' * R)
        end

        Rnorm = norm(R, 2)
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)

        # Lock converged vectors
        for i in 1:Nlow
            if !converged[i] && norm(R[:, i]) < thresh
                converged[i] = true
                V_lock = hcat(V_lock, X[:, i])
            end
        end

        if all(converged)
            println("converged!")
            return (Σ, X)
        end

        # Preconditioning
        t = zero(similar(R))
        for i = 1:Nlow
            if !converged[i]
                C = 1.0 ./ (Σ[i] .- D)
                t[:, i] = C .* R[:, i]
            end
        end

        # Subspace expansion
        if size(V, 2) <= Naux - Nlow
            V = hcat(V, t)
        else
            V = hcat(X, t)
        end
    end
end

main("Si")
