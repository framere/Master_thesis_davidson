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


function davidson(A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    l_buffer::Integer,
    thresh::Float64,
    system::String = ""
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    n_b = size(V, 2)
    nu_0 = max(l_buffer, n_b)
    nevf = 0

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)

    iter = 0

    while nevf < l_buffer
        iter += 1

        # Orthogonalize V against locked vectors
        if size(V_lock, 2) > 0
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    V[:, j] -= v_lock * (v_lock' * V[:, j])
                end
            end
        end
        V = Matrix(qr(V).Q)

        # Rayleigh-Ritz
        H = Hermitian(V' * (A * V))
        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)
        X = V * U  # Ritz vectors

        # Compute residuals like you originally did
        R = X .* Σ' - A * X
        norms = vec(norm.(eachcol(R)))

        conv_indices = Int[]
        for i = 1:size(R, 2)
            if norms[i] <= thresh
                push!(conv_indices, i)
                push!(Eigenvalues, Σ[i])
                Ritz_vecs = hcat(Ritz_vecs, X[:, i])
                V_lock = hcat(V_lock, X[:, i])
                nevf += 1
                println(@sprintf("Converged eigenvalue %.10f with norm %.2e (EV %d)", Σ[i], norms[i], nevf))
                if nevf >= l
                    println("Converged all eigenvalues.")
                    return (Eigenvalues, Ritz_vecs)
                end
            end
        end

        non_conv_indices = setdiff(1:size(R, 2), conv_indices)
        X_nc = X[:, non_conv_indices]
        Σ_nc = Σ[non_conv_indices]
        R_nc = R[:, non_conv_indices]

        # Correction vectors
        t = Matrix{T}(undef, size(A, 1), length(non_conv_indices))
        ϵ = 1e-6  # small value to avoid division by zero
        for (i, idx) in enumerate(non_conv_indices)
            denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
            t[:, i] = R_nc[:, i] ./ denom
        end

        # Orthogonalize corrections
        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-10)
        # println("→ Requested $(size(t, 2)) corrections, kept $n_b_hat after orthogonalization")

        # Update subspace
        if size(V, 2) + n_b_hat > n_aux|| length(conv_indices) > 0 || n_b_hat == 0 
            V = hcat(X_nc, T_hat)
            n_b = size(V, 2)
        else
            V = hcat(V, T_hat)
            n_b += n_b_hat
        end

        println("Iter $iter: V_size = $n_b, Converged = $nevf, Norm = $(minimum(norms))")
    end

    return (Eigenvalues, Ritz_vecs)
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

    # read the matrix
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
    # the two test systems He and hBN are hardcoded
    system = system
    
    Nlow = 16 # we are interested in the first Nlow eigenvalues
    Naux = Nlow * 16 # let our auxiliary space be larger (but not too large)
    l = 200 # number of eigenvalues to compute
    l_buffer = 210 # number of eigenvalues to compute before checking convergence


    # read the matrix
    A = load_matrix(system)
    N = size(A, 1)

    ## initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # # initial guess vectors (stochastic guess)
    # V = rand(N,Nlow) .- 0.5

    # perform Davidson algorithm
    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, l, l_buffer, 1e-4, system)


    # sort
    idx = sortperm(Σ)
    Σ = Σ[idx] # sort eigenvalues
    U = U[:,idx] # sort the converged eigenvectors

    # perform exact diagonalization as a reference
    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A) 

    # display("text/plain", Σexact[1:l]')
    # display("text/plain", Σ')
    display("text/plain", (Σ - Σexact[1:l])')
end


main("hBN") # or "hBN", "Si"