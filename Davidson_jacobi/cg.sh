#!/bin/sh
#SBATCH -J cg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p andoria
#SBATCH --time=3-00:00:00

JULIA=/home/fmereto/julia/bin/julia

$JULIA dav_jac_cg.jl

