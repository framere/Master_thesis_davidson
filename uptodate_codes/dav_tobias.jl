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
    
    Nlow = 100 # we are interested in the first Nlow eigenvalues
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
    @time Σ, U = davidson(A, V, Naux, 1e-5, system)

    # perform exact diagonalization as a reference
    #println("Full diagonalization")
    #@time Σexact, Uexact = eigen(A) 

    #display("text/plain", Σexact[1:Nlow]')
    #display("text/plain", Σ')
    #display(A("text/plain", (Σ-Σexact[1:Nlow])')
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
    logfile = open("davidson_log_$system.txt", "w") 
    # iterations
    iter = 0
    nevf = 0 # number of converged eigenvalues
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
        
        n_converg = 0
        norms = zeros(size(R,2))
        for i = 1:size(R,2)
            norms[i] = norm(R[:,i])
            if norm(R[:,i]) < thresh
                n_converg += 1
            end
        end

        # status output
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        @printf(logfile, "%d %.6e\n", iter, Rnorm)
        print(output)
        
        if Rnorm < thresh
            println("converged!")
            close(logfile)
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


# systems = ["hBN", "Si"] #"He",

# for system in systems
#      println("system: ", system)
#      main(system)
# end



main("Si")
