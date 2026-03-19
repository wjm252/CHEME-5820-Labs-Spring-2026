using VLDataScienceMachineLearningPackage
using Statistics
using JLD2
using LinearAlgebra
using Plots
using Distances
using NNlib
using Distributions
using PrettyTables
using DataFrames
using StatsBase
using Random
using IJulia

include(joinpath(@__DIR__, "src", "BagOfWords.jl"))
include(joinpath(@__DIR__, "src", "CBOW.jl"))
include(joinpath(@__DIR__, "src", "SkipGram.jl"))
include(joinpath(@__DIR__, "src", "NegativeSampling.jl"))
include(joinpath(@__DIR__, "src", "GloVe.jl"))
