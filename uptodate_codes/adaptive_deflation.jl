using LinearAlgebra
using Printf
using JLD2

include("functions_davidson.jl")

using LinearAlgebra
using Printf

function davidson(A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    deg_thresh::Float64 = 1e-3,  # Threshold for considering eigenvalues degenerate
    system::String = ""
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    n_b = size(V, 2)
    nu_0 = max(l, n_b)
    nevf = 0

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)

    iter = 0

    while nevf < l
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

        # Compute residuals
        R = X .* Σ' - A * X
        norms = vec(norm.(eachcol(R)))

        # Find all indices that meet the convergence threshold
        potential_conv_indices = findall(norms .<= thresh)
        
        # Group degenerate eigenvalues
        degenerate_groups = Vector{Vector{Int}}()
        remaining_indices = collect(1:length(Σ))
        
        while !isempty(remaining_indices)
            current_idx = popfirst!(remaining_indices)
            group = [current_idx]
            
            # Find all eigenvalues close to the current one
            for (i, idx) in enumerate(remaining_indices)
                if abs(Σ[idx] - Σ[current_idx]) < deg_thresh
                    push!(group, idx)
                end
            end
            
            # Remove found indices from remaining
            filter!(x -> !(x in group[2:end]), remaining_indices)
            push!(degenerate_groups, group)
        end

        # Determine which vectors to lock
        conv_indices = Int[]
        for group in degenerate_groups
            # Check if all in group are converged
            all_converged = all(in(potential_conv_indices), group)
            
            if all_converged
                append!(conv_indices, group)
                append!(Eigenvalues, Σ[group])
                Ritz_vecs = hcat(Ritz_vecs, X[:, group])
                V_lock = hcat(V_lock, X[:, group])
                nevf += length(group)
                
                for idx in group
                    println(@sprintf("Converged eigenvalue %.10f with norm %.2e (EV %d)", Σ[idx], norms[idx], nevf - length(group) + findfirst(==(idx), group)))
                end
                
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

        # Update subspace
        if size(V, 2) + n_b_hat > n_aux || length(conv_indices) > 0 || n_b_hat == 0 
            max_new_vectors = n_aux - size(X_nc, 2)  # Space left after keeping X_nc
            T_hat = T_hat[:, 1:min(n_b_hat, max_new_vectors)]  # Truncate if needed
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

function load_eigenresults(output_file="eigen_results.jld2")
    # Unpack directly into variables
    data = load(output_file)  # Returns a Dict-like object
    Σexact = data["Σexact"]  # Access by key
    Uexact = data["Uexact"]
    return Σexact, Uexact
end

function main(system::String, l::Integer = 200)
    # the two test systems He and hBN are hardcoded
    system = system
    filename = "../../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat" # personal
    # filename = filename = "../Davidson_algorithm/m_pp_" * system * ".dat"
    
    Nlow = 16 # Starting dimension for the subspace
    Naux = Nlow * 16 # let our auxiliary space be larger (but not too large)
    # l = 200 # number of eigenvalues to compute
    
    # read the matrix
    A = load_matrix(system, filename)
    N = size(A, 1)

    ## initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # perform Davidson algorithm
    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, l, 1e-2, 7e-2, system)


    # sort
    idx = sortperm(Σ)
    Σ = Σ[idx] # sort eigenvalues
    U = U[:,idx] # sort the converged eigenvectors

    # Perform exact diagonalization as reference
    println("\nPerforming full diagonalization for reference...")
    # Σexact, Uexact = load_eigenresults("../../MA_best/Eigenvalues_folder/eigen_results_$system.jld2")
    Σexact, Uexact = load_eigenresults("../../Final_codes_MA/eigen_results_$system.jld2")
    

    # Display difference
    println("\nDifference between Davidson and exact eigenvalues:")
    display("text/plain", (Σ - Σexact[1:l])')
end


ls = [216] #, 288, 360

for l in ls
    println("Running for Nlow = $l")
    main("hBN", l)
end