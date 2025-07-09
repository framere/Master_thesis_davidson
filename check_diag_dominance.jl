using LinearAlgebra
using Printf

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

function analyze_diagonal_dominance(A::AbstractMatrix{T}, output_filename::String) where T<:Number
    N = size(A, 1)
    
    # Open output file
    output_file = open(output_filename, "w")
    
    count_non_diago_dominant_rows = 0
    for i in 1:N
        diag_element = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in 1:N if j != i)

        # Write to file
        @printf(output_file, "%.15e %.15e\n", diag_element, off_diag_sum)

        if diag_element <= off_diag_sum
            count_non_diago_dominant_rows += 1            
        end
    end
    
    close(output_file)
    
    return count_non_diago_dominant_rows
end

function main(system::String)
    filename = "../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat" # personal
    output_filename = "diagonal_analysis_" * system * ".txt"

    A = load_matrix(system, filename)

    non_dominant_count = analyze_diagonal_dominance(A, output_filename)
    if non_dominant_count > 0
        println("Matrix is not diagonally dominant in $non_dominant_count rows.")
        println("Results written to $output_filename")
    else
        println("Matrix is diagonally dominant in all rows.")
        println("Results written to $output_filename")
    end
end

systems = ["He", "hBN", "Si"] 

for system in systems
    main(system)
end