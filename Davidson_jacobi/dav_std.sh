#!/bin/sh
#SBATCH -J std_block
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p andoria
#SBATCH --time=3-00:00:00

JULIA=/home/fmereto/julia/bin/julia

$JULIA /home/fmereto/Master_arbeit/uptodate_codes/dav_tobias.jl

