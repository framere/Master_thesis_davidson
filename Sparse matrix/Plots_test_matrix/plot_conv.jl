using Plots
using DelimitedFiles

# for i in collect(10:10:100)
#     # Read the data from the file
#     lanczos = readdlm("Lanczos_log_factor_$i.txt")
#     davidson = readdlm("davidson_test_factor_$i.txt")
#     # Extract the first column (iteration number) and second column (Rnorm)
#     iter_l = lanczos[:, 1]
#     Rnorm_l = lanczos[:, 2]
#     iter_d = davidson[:, 1]
#     Rnorm_d = davidson[:, 2]
    
#     # Create a plot for the standard Davidson algorithm
#     plot(iter_l, Rnorm_l, yscale = :log10, label="Lanczos factor $i", title="Lanczos Convergence (factor $i)", xlabel="Iteration", ylabel="Rnorm", legend=:topright)
#     plot!(iter_d, Rnorm_d, label="Davidson factor $i")
#     # Save the plot to a file
#     savefig("lanczos_vs_davidson_factor_$i.pdf")
# end

lanczos = readdlm("Lanczos_log_factor_300.txt")
davidson = readdlm("davidson_test_factor_300.txt")

# Extract the first column (iteration number) and second column (Rnorm)
iter_l = lanczos[:, 1]
Rnorm_l = lanczos[:, 2]
iter_d = davidson[:, 1]
Rnorm_d = davidson[:, 2]

# Create a plot for the standard Davidson algorithm
plot(iter_l, Rnorm_l, yscale = :log10, label="Lanczos factor 300", title="Lanczos Convergence (factor 300)", xlabel="Iteration", ylabel="Rnorm", legend=:topright)
plot!(iter_d, Rnorm_d, label="Davidson factor 300")
savefig("lanczos_vs_davidson_factor_300.pdf")