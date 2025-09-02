using LinearAlgebra
using Printf
using JLD2
using IterativeSolvers
using LinearMaps

function load_eigenresults(output_file="eigen_results.jld2")
    # Unpack directly into variables
    data = load(output_file)  # Returns a Dict-like object
    Σexact = data["Σexact"]  # Access by key
    Uexact = data["Uexact"]
    return Σexact, Uexact
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
    # filename = "../Davidson_algorithm/m_pp_" * system * ".dat"
    filename = "../../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat" # personal
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A  # for largest eigenvalues of original matrix
    return Hermitian(A)
end

function correction_equations_cg(A, U, lambdas, R; tol=1e-2, maxiter=100)
    n, k = size(U)
    S = zeros(eltype(A), n, k)

    for j in 1:k
        u = U[:, j]
        λ = lambdas[j]
        r = R[:, j]

        # Define operator M_j(s) = (I - U U^*)(A - λ I)(I - U U^*) s
        M_apply = function(x)
            # Project x onto the orthogonal complement of U
            x_perp = x - U * (U' * x)
            # Apply (A - λI) to the projected vector
            tmp = (A - λ * I) * x_perp
            # Project the result back onto the orthogonal complement
            return tmp - U * (U' * tmp)
        end

        # Create LinearMap object
        M_op = LinearMap{eltype(A)}(M_apply, n, n; ismutating=false, ishermitian=true)

        # Right-hand side: project residual onto orthogonal complement
        rhs = r - U * (U' * r)
        rhs = -rhs  # We want to solve M_j(s) = -r_perp

        # Solve using Conjugate Gradient
        s_j = cg(M_op, rhs; abstol=tol, maxiter=maxiter)

        # Ensure strict orthogonality
        s_j = s_j - U * (U' * s_j)

        S[:, j] = s_j
    end

    return S
end

function correction_equations_minres(A, U, lambdas, R; tol=1e-2, maxiter=100)
    n, k = size(U)
    S = zeros(eltype(A), n, k)

    for j in 1:k
        u = U[:, j]
        λ = lambdas[j]
        r = R[:, j]

        # Define operator M_j(s) = (I - U U^*)(A - λ I)(I - U U^*) s
        M_apply = function(x)
            # Project x onto the orthogonal complement of U
            x_perp = x - U * (U' * x)
            # Apply (A - λI) to the projected vector
            tmp = (A - λ * I) * x_perp
            # Project the result back onto the orthogonal complement
            return tmp - U * (U' * tmp)
        end

        # Create LinearMap object
        M_op = LinearMap{eltype(A)}(M_apply, n, n; ismutating=false, ishermitian=true)

        # Right-hand side: project residual onto orthogonal complement
        rhs = r - U * (U' * r)
        rhs = -rhs  # We want to solve M_j(s) = -r_perp

        # Solve using MINRES
        s_j = minres(M_op, rhs; abstol=tol, maxiter=maxiter)

        # Ensure strict orthogonality
        s_j = s_j - U * (U' * s_j)

        S[:, j] = s_j
    end

    return S
end

# a simple implementation of the block Davidson method for a Hermitian matrix A
function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    system::String;
    solver::Symbol = :cg,  # Choose between :cg and :minres
    max_iter = 50
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # iterations
    iter = 0
    while true
        iter = iter + 1
        
        # orthogonalize guess orbitals (using QR decomposition)
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)
        
        # construct and diagonalize Rayleigh matrix
        H = Hermitian(V' * (A * V))
        Σ, U = eigen(H)
        Σ = Σ[1:Nlow]
        U = U[:, 1:Nlow]
        X = V * U # Ritz vectors
        

        if iter > max_iter
            println("Reached maximum iterations ($max_iter) without convergence.")
            return (Σ, X)  # Return the best found so far
        end
        
        R = A * X - X * Diagonal(Σ) # residual vectors
        Rnorm = norm(R) # Frobenius norm
      
        # status output
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)
        
        if Rnorm < thresh
            println("converged!")
            return (Σ, X)
        end
        
        # Solve correction equations using chosen solver
        if solver == :cg
            t = correction_equations_cg(A, X, Σ, R; tol=1e-2, maxiter=100)
        elseif solver == :minres
            t = correction_equations_minres(A, X, Σ, R; tol=1e-2, maxiter=100)
        else
            error("Unknown solver: $solver. Choose :cg or :minres")
        end
        
        # update guess basis
        if size(V,2) + size(t,2) <= Naux
            V = hcat(V, t) # concatenate V and correction
        else
            # Restart: keep only the current Ritz vectors and new corrections
            V = hcat(X, t)
        end
        
        # Optional: limit the basis size to prevent excessive memory usage
        if size(V,2) > Naux
            # Keep the most recent vectors
            V = V[:, end-Naux+1:end]
        end
    end
end

function main(system::String, Nlow::Int; solver::Symbol = :cg)
    system = system
    
    Naux = Nlow * 7 # let our auxiliary space be larger (but not too large)

    # read the matrix
    A = load_matrix(system)
    N = size(A, 1)

    # initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # perform Davidson algorithm
    println("Davidson with $(solver == :cg ? "CG" : "MINRES") solver")
    @time Σ, U = davidson(A, V, Naux, 1e-2, system, solver=solver)

    # perform exact diagonalization as a reference
    println("Full diagonalization")
    Σexact, Uexact = load_eigenresults("../eigen_results_"*system*".jld2") 

    display("text/plain", Σ')
    display("text/plain", (Σ-Σexact[1:Nlow])')
end

N_lows = [16]
molecules = ["He", "Si", "hBN"]
for molecule in molecules
    for Nlow in N_lows
        println("Running for Nlow = $Nlow")
        # Try both solvers
        println("Using CG solver:")
        main(molecule, Nlow, solver=:cg)

        println("\nUsing MINRES solver:")
        main(molecule, Nlow, solver=:minres)
    end
end