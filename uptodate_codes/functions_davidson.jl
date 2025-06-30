using LinearAlgebra
using Printf

# Orthogonalize correction vectors against current and locked vectors
function select_corrections_ORTHO(t_candidates, V, V_lock, η, droptol; maxorth=2)
    ν = size(t_candidates, 2)
    n_b = 0
    T_hat = Matrix{eltype(t_candidates)}(undef, size(t_candidates, 1), ν)

    for i in 1:ν
        t_i = t_candidates[:, i]
        old_norm = norm(t_i)
        k = 0

        while k < maxorth
            k += 1

            for j in 1:size(V, 2)
                t_i -= V[:, j] * (V[:, j]' * t_i)
            end

            # # I don't think this is needed, as V_lock is a subset of V
            # for j in 1:size(V_lock, 2)
            #     t_i -= V_lock[:, j] * (V_lock[:, j]' * t_i)
            # end

            new_norm = norm(t_i)
            if new_norm > η * old_norm
                break
            end
            old_norm = new_norm
        end

        if norm(t_i) > droptol
            n_b += 1
            T_hat[:, n_b] = t_i / norm(t_i)
        end
    end

    return T_hat[:, 1:n_b], n_b
end

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

function orthogonalize(V::Matrix, against::Matrix)
    if size(against, 2) == 0
        return Matrix(qr(V).Q)
    end
    for i in 1:size(against, 2)
        v = against[:, i]
        for j in 1:size(V, 2)
            V[:, j] -= v * (v' * V[:, j])
        end
    end
    return Matrix(qr(V).Q)
end


function rayleigh_ritz_projection(
    A::Hermitian{T, Matrix{T}}, 
    V::Matrix{T}, 
    nev::Int
)::Tuple{Vector{T}, Matrix{T}, Matrix{T}} where T<:Number

    H = Hermitian(V' * A * V)
    Σ, U = eigen(H, 1:nev)
    X = V * U
    R = X .* Σ' .- A * X
    return Σ, X, R
end