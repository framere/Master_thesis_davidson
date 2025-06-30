using IterativeSolvers
using LinearAlgebra

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


A = sparse_matrix(100, 300)  # Example sparse matrix of size 100x100
# Right-hand side vector
b = rand(100)  # Random vector of size 100
b = b / norm(b)  # Normalize the vector

# Jacobi (diagonal) preconditioner
D_inv = Diagonal(1.0 ./ diag(A))  # Pl ≈ A⁻¹ (approximately)

# Solve using GMRES with left preconditioning
x, history = gmres(A, b; Pl=D_inv, log=true, reltol=1e-8)

#exact solution
exact_solution = A \ b

# check how close the solution is to the exact one
residual = norm(A * x - b)
println("Residual norm: ", residual)
# Display results
println("Approximate solution with GMRES + preconditioning: x = ", x)
println("Exact solution: x_exact = ", exact_solution)
