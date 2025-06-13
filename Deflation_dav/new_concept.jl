using LinearAlgebra
using Printf

# Orthogonalize correction vectors against current basis and locked (converged) vectors
function select_corrections_ORTHO(t_candidates, V, V_lock, η, droptol; maxorth=2)
    ν = size(t_candidates, 2)
    n_b = 0
    T_hat = Matrix{eltype(V)}(undef, size(V, 1), ν)

    for i in 1:ν
        t_i = t_candidates[:, i]
        old_norm = norm(t_i)
        k = 0

        while k < maxorth
            k += 1

            for j in 1:size(V, 2)
                t_i -= V[:, j] * (V[:, j]' * t_i)
            end
            for j in 1:size(V_lock, 2)
                t_i -= V_lock[:, j] * (V_lock[:, j]' * t_i)
            end

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

# Main Davidson method
function davidson(A::AbstractMatrix{T},
                  V::Matrix{T},
                  n_aux::Integer,
                  l::Integer,
                  thresh::Float64,
                  system::String = ""
                 )::Tuple{Vector{T}, Matrix{T}} where T<:Number

    n_b = size(V, 2)
    nu_0 = max(1, n_b)
    nevf = 0

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)  # Converged eigenvectors

    iter = 0

    while nevf < l
        iter += 1

        # Orthogonalize V
        V = Matrix(qr(V).Q)

        # Project and solve small eigenproblem
        H = Hermitian(V' * (A * V))
        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)
        X = V * U

        # Compute residuals
        R = A * X - X * Diagonal(Σ)

        norms = vec(norm.(eachcol(R)))
        conv_indices = findall(x -> x <= thresh, norms)

        for i in conv_indices
            push!(Eigenvalues, Σ[i])
            Ritz_vecs = hcat(Ritz_vecs, X[:, i])
            V_lock = hcat(V_lock, X[:, i])
            nevf += 1
            println(@sprintf("Converged eigenvalue %.10f with residual %.2e (EV %d)", Σ[i], norms[i], nevf))
            if nevf >= l
                println("All eigenvalues converged.")
                return (Eigenvalues, Ritz_vecs)
            end
        end

        non_conv_indices = setdiff(1:nu, conv_indices)
        X_nc = X[:, non_conv_indices]
        Σ_nc = Σ[non_conv_indices]
        R_nc = R[:, non_conv_indices]

        # Correction vectors with diagonal preconditioner
        t = Matrix{T}(undef, size(A,1), length(non_conv_indices))
        ϵ = 1e-10

        for (i, idx) in enumerate(non_conv_indices)
            denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
            t[:, i] = R_nc[:, i] ./ denom
        end

        # Orthonormalize corrections
        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-6)

        # Restart logic
        if size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            V = hcat(X_nc, T_hat)
        else
            V = hcat(V, T_hat)
        end

        println("Iteration $iter: Subspace size = $(size(V,2)), Converged = $nevf")
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
    
    Nlow =16 # we are interested in the first Nlow eigenvalues
    Naux = Nlow * 16 # let our auxiliary space be larger (but not too large)

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
    @time Σ, U = davidson(A, V, Naux, 50, 1e-5, system)

    # perform exact diagonalization as a reference
    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A) 

    display("text/plain", Σexact[1:Nlow]')
    display("text/plain", Σ')
    display("text/plain", (Σ - Σexact[1:Nlow])')
end


main("He") # or "hBN", "Si"