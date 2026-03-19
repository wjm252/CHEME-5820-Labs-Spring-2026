# setup paths -
const _ROOT = pwd();
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# check do we have a Manifest.toml file?
using Pkg;
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
     Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

using VLDataScienceMachineLearningPackage
using Statistics
using JLD2
using LinearAlgebra
using Plots
using Colors
using Distances
using NNlib
using Distributions
using PrettyTables
using DataFrames
using StatsBase
using Random

# include my codes -
include(joinpath(_PATH_TO_SRC, "BagOfWords.jl"));
include(joinpath(_PATH_TO_SRC, "TFIDF.jl"));
include(joinpath(_PATH_TO_SRC, "PMI.jl"));
include(joinpath(_PATH_TO_SRC, "CBOW.jl"));