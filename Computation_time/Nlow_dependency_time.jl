using LinearAlgebra
using Printf

function sparse_matrix(N::Int, factor::Int)
    A = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i, j] = rand() 
            else
                A[i, j] = rand() / factor
            end
        end
    end
    A = 0.5 * (A + A')

    return Hermitian(A)
end

function main(Nlow::Int)
    N = 1000
    Naux = Nlow * 2
    Nlow = Nlow
    factor = 300
    thresh = 1e-5
   
    # create a sparse matrix
    A = sparse_matrix(N, factor)
    
    # initial guess for eigenvectors
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # run Davidson algorithm
    println("Davidson")
    t_start = time()
    @time iterations = davidson(A, V, Naux, thresh, Nlow)
    t_end = time()
    elapsed_time = t_end - t_start

    return (iterations, elapsed_time)
end

function davidson(A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    Nlow::Integer
    )::Integer where T<:Real
    
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)

    iter = 0
    while true
        iter += 1
    
        # orthogonalize guess orbitals (using QR decomposition)
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # construct and diagonalize Rayleigh matrix
        H = Hermitian(V'*(A*V))
        Σ, U = eigen(H, 1:Nlow)

        X = V*U # Ritz vecors
        R = X.*Σ' - A*X # residual vectors
        Rnorm = norm(R,2) # Frobenius norm
        
        # status output
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)
        
        if Rnorm < thresh
            println("converged!")
            return (iter)
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

logfile = open("Computation_time/data_iterations_Nlow.txt", "w")
for Nlow in collect(2:4:32)
    println("Nlow = ", Nlow)
    iterations, elapsed_time = main(Nlow)
    @printf(logfile, "%d % d% .6e\n", Nlow, iterations, elapsed_time)
end
close(logfile)