using LinearAlgebra
using Printf
using JLD2
using BenchmarkTools

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
    # filename = "../../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat" # personal
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A  # for largest eigenvalues of original matrix
    return Hermitian(A)
end

function load_eigenresults(output_file="eigen_results.jld2")
    # Unpack directly into variables
    data = load(output_file)  # Returns a Dict-like object
    Σexact = data["Σexact"]  # Access by key
    Uexact = data["Uexact"]
    return Σexact, Uexact
end


function main(system::String, Nlow::Int)
    # the two test systems He and hBN are hardcoded
    system = system
    
    # Nlow = 200 # we are interested in the first Nlow eigenvalues
    Naux = Nlow * 7 # let our auxiliary space be larger (but not too large)

    # read the matrix
    A = load_matrix(system)
    N = size(A, 1)

    # initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # # initial guess vectors (stochastic guess)
    # V = rand(N,Nlow) .- 0.995

    # perform Davidson algorithm
    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, 1e-2, system)

    # perform exact diagonalization as a reference
    println("Full diagonalization")
    Σexact, Uexact = load_eigenresults("../EV_calculation/eigen_results_"*system*".jld2") 

    #display("text/plain", Σexact[1:Nlow]')
    display("text/plain", Σ')
    # display("text/plain", U)
    display("text/plain", (Σ-Σexact[1:Nlow])')
end


# a simple implementation of the block Davidson method for a Hermitian matrix A
function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    system::String # default system is hBN
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # diagonal part of A (for preconditioner)
    D = diag(A)
    # logfile = open("davidson_log_$system.txt", "w") 
    # iterations
    iter = 0
    while true
        iter = iter + 1
        
        # orthogonalize guess orbitals (using QR decomposition)
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # construct and diagonalize Rayleigh matrix
        H = Hermitian(V'*(A*V))
        Σ, U = eigen(H, 1:Nlow)

        X = V*U # Ritz vectors
        R = X.*Σ' - A*X # residual vectors
        Rnorm = norm(R,2) # Frobenius norm
      
        # status output
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        # @printf(logfile, "%d %.6e\n", iter, Rnorm)
        print(output)
        
        if Rnorm < thresh
            println("converged!")
            # close(logfile)
            return (Σ, X)
        end

        # update guess space using diagonal preconditioner 
        t = zero(similar(R)) 
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D) 
            t[:,i] = C .* R[:,i] # the new basis vectors
        end

        # update guess basis
        if size(V,2) <= Naux-Nlow
            V = hcat(V,t) # concatenate V and t
        else
            V = hcat(X,t) # concatenate X and t 
        end
    end
end

N_lows = [16, 30, 60, 90]
molecules = ["He", "hBN", "Si"]
for Nlow in N_lows
    println("Running for Nlow = $Nlow")
    for molecule in molecules
        @btime main($molecule, $Nlow)
    end
end
