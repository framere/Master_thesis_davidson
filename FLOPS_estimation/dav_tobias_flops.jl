using LinearAlgebra
using Printf

# Global FLOP counter
global NFLOPs = 0

function count_matmul_flops(M::Int, N::Int, K::Int)
    global NFLOPs += 2 * M * N * K
end

function count_diag_flops(N::Int)
    global NFLOPs += 20 * N^3
end

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
    A = -A
    return Hermitian(A)
end

function main(system::String, Nlow::Int)
    global NFLOPs
    NFLOPs = 0  # Reset FLOP counter
    
    Naux = Nlow * 2
    A = load_matrix(system)
    N = size(A, 1)

    # initial guess (naiv)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, 1e-2, system)

    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A)
    count_diag_flops(N)

    display("text/plain", Σ')
    display("text/plain", (Σ - Σexact[1:Nlow])')

    println("Total estimated FLOPs: $(NFLOPs)")
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    system::String
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    global NFLOPs

    Nlow = size(V, 2)
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)
    iter = 0

    while true
        iter += 1

        # QR-Orthogonalisierung
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # Rayleigh-Matrix: H = V' * (A * V)
        temp = A * V
        count_matmul_flops(size(A,1), size(A,2), size(V,2))  # A*V
        H = V' * temp
        count_matmul_flops(size(V,2), size(V,1), size(temp,2))  # V'*temp

        H = Hermitian(H)
        Σ, U = eigen(H, 1:Nlow)
        count_diag_flops(size(H,1))  # kleine Diagonalisierung

        X = V * U
        count_matmul_flops(size(V,1), size(V,2), size(U,2))  # V*U

        # R = X*Σ' - A*X
        R = X .* Σ'  # Skalierung
        temp2 = A * X
        count_matmul_flops(size(A,1), size(A,2), size(X,2))  # A*X
        R .-= temp2

        Rnorm = norm(R, 2)

        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)

        if Rnorm < thresh
            println("converged!")
            return (Σ, X)
        end

        # Preconditioning
        t = similar(R)
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D)
            t[:,i] = C .* R[:,i]
        end

        # Update V
        if size(V,2) <= Naux - Nlow
            V = hcat(V, t)
        else
            V = hcat(X, t)
        end
    end
end

N_lows = [216, 288, 360]

for Nlow in N_lows
    println("Running for Nlow = $Nlow")
    main("hBN", Nlow)
end
