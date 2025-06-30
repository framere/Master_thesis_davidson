using LinearAlgebra

function jacobi_filter(A::AbstractMatrix{Float64}, V::Matrix{Float64}, n::Int, tol::Float64; Nblock::Int=5)
    V = qr(V).Q                     # Orthonormalize V
    k = 0                           # Index of first unconverged eigenvector
    λ = zeros(n)
    println("Starting Jacobi-Davidson with $n eigenvalues...")
    iter = 0
    while k < n
        println("Iteration: ", iter + 1, ", k = ", k + 1, " / ", n)
        iter += 1
        block_end = min(k + Nblock - 1, n - 1)
        block = k+1:block_end+1     # Julia uses 1-based indexing

        Vblock = V[:, block]
        λblock = zeros(length(block))
        R = similar(Vblock)         # Residuals

        for (j, i) in enumerate(block)
            vi = V[:, i]
            λi = dot(vi, A * vi)
            λblock[j] = λi
            R[:, j] = A * vi - λi * vi
        end

        # Solve correction equations approximately
        S = similar(R)
        for (j, i) in enumerate(block)
            vi = V[:, i]
            λi = λblock[j]
            M_inv = Diagonal(1.0 ./ (diag(A) .- λi))
            s = M_inv * R[:, j]  # crude approximate solve
            S[:, j] = s
        end

        # Form W and orthogonalize
        W = hcat(V[:, block], S)
        if k > 0
            Vconv = V[:, 1:k]
            for i in 1:size(W, 2)
                W[:, i] -= Vconv * (Vconv' * W[:, i])
            end
        end
        W = qr(W).Q

        # Project and diagonalize
        Ã = W' * A * W
        D, Q = eigen(Symmetric(Ã))
        V[:, block] .= W * Q[:, 1:length(block)]
        λ[block] .= D[1:length(block)]

        # Check convergence
        for i in block
            ri = A * V[:, i] - λ[i] * V[:, i]
            if norm(ri) > tol
                k = i
                break
            else
                k = i + 1
            end
        end
    end

    return V[:, 1:n], λ[1:n]
end

# --- Test it below ---
n = 1000

# function sparse_matrix(N::Int, factor::Int)
#     A = zeros(Float64, N, N)
#     for i in 1:N
#         for j in 1:N
#             if i == j
#                 A[i, j] = rand() 
#             else
#                 A[i, j] = rand() / factor
#             end
#         end
#     end
#     A = 0.5 * (A + A')

#     return Hermitian(A)
# end

function load_matrix(system::String,
    filename::String 
    )
    if system == "He"
        N = 4488
    elseif system == "hBN"
        N = 6863        
    elseif system == "Si"
        N = 6201
    else
        error("Unknown system: $system")
    end

    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A  # for largest eigenvalues of original matrix
    return Hermitian(A)
end

system = "He"  # or "hBN", "Si"
filename = "../Davidson_algorithm/m_pp_" * system * ".dat" # replace with your matrix file

A = load_matrix(system, filename)
N = size(A, 1)

n = 10 
V0 = rand(N,n) .- 0.5

@time V, λ = jacobi_filter(A, V0, n, 1e-6)
println("Computed eigenvalues: ", λ)

# compare with exact eigenvalues
@time exact_λ = eigen(A).values
println("Exact eigenvalues: ", exact_λ[1:n])
println("Difference: ", norm(λ - exact_λ[1:n]))