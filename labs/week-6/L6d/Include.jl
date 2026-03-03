# setup paths -
const _ROOT = @__DIR__

# load external packages -
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# using statements -
using VLDataScienceMachineLearningPackage
using Images
using ImageInTerminal
using FileIO
using ImageIO
using OneHotArrays
using Statistics
using JLD2
using LinearAlgebra
using Plots
using Colors
using Distances
using DataStructures
using Test
using Random
using LinearAlgebra
using Printf
using NNlib
using DataFrames
using StatsPlots
using StatsBase
using Distributions
using PrettyTables

# set the random seed for reproducibility
Random.seed!(1234);