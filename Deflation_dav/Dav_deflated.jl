using LinearAlgebra
using Printf

function main(system::String)
    # the two test systems He and hBN are hardcoded
    system = system
    
    Nlow = 16 # we are interested in the first Nlow eigenvalues
    Naux = Nlow * 16 # let our auxiliary space be larger (but not too large)

    if system == "He"
        N = 4488
    elseif system == "hBN"
        N = 6863        
    elseif system == "Si"
        N = 6201
    else
        println("Systen ", system, " unknown.")
        exit()
    end

    # read the matrix
    filename = "../Davidson_algorithm/m_pp_" * system * ".dat"
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N*N)
    read!(file, A)
    close(file)
    A = reshape(A, N, N)
    A = -A # because we are interested in the largest eigenvalues
    A = Hermitian(A)

    # initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # perform Davidson algorithm
    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, 1e-5, N, system)

    # perform exact diagonalization as a reference
    println("Full diagonalization")
    @time Σexact, Uexact = eigen(A) 

    display("text/plain", Σexact[1:Nlow]')
    display("text/plain", Σ')
    display("text/plain", (Σ-Σexact[1:Nlow])')
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    N ::Integer,
    system::String # default system is hBN
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    nevf = 0 # number of converged eigenvalues

    # diagonal part of A (for preconditioner)
    D = diag(A)
    
    iter = 0 # iteration counter

    Ritz_vecs = [] # Ritz vectors
    Eigenvalues = Float64[] # Ritz eigenvalues
    while true
        iter = iter + 1
        
        if nevf > 0
            Xconv = hcat(Ritz_vecs...) # converged eigenvectors
            # V -= Xconv * (Xconv' * V)
            for j in 1:size(V,2)
                V[:, j] -= Xconv * (Xconv' * V[:, j])  # project out component in span(Xconv) --> Is it correct? Are the ritz vectors orthogonalized already?
            end
        end

        # orthogonalize guess orbitals (using QR decomposition)
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)
        

        # construct and diagonalize Rayleigh matrix
        H = Hermitian(V'*(A*V))
        nu = min(size(H, 2), Nlow-nevf) # number of new basis vectors
        Σ, U = eigen(H, 1:nu)

        X = V*U # Ritz vecors
        R = X.*Σ' - A*X # residual vectors
        # Rnorm = norm(R,2) # Frobenius norm
        
        n_converg = 0
        norms = zeros(size(R,2))
        for i = 1:size(R,2)
            norms[i] = norm(R[:,i])
            if norm(R[:,i]) < thresh
                n_converg += 1
                push!(Ritz_vecs, X[:,i])
                push!(Eigenvalues, Σ[i])
                println("converged eigenvalue ", Σ[i], " with norm ", norms[i])
            end
        end
        
        nevf += n_converg # number of converged eigenvalues

        output = @sprintf("iter=%6d  nevf= %6d  min_norm=%11.3e  size(V,2)=%6d\n", iter, nevf, minimum(norms), size(V,2))
        print(output)

        if nevf >= Nlow
            println("converged!")
            idx = sortperm(Eigenvalues)
            Eigenvalues = Eigenvalues[idx] # they are not sorted! 
            return (Eigenvalues, Xconv)
        end

        # update guess space using diagonal preconditioner 
        t = zero(similar(R)) 
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D) 
            t[:,i] = C .* R[:,i] # the new basis vectors
        end

        V0 = zeros(N, Nlow)
        for i = 1:Nlow
            V0[i,i] = 1.0
        end

        # update guess basis
        if size(V,2) <= Naux-Nlow && n_converg == 0 # could be wrong as Naus-Nlow could not be representative anymore
            V = hcat(V,t) # concatenate V and t
        else
            V = hcat(V0, X[:,n_converg +1 : end],t) # concatenate X and t 
        end
    end
end

main("He")