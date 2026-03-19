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
using JLD2
using LinearAlgebra
using Plots
using NNlib
using PrettyTables
using StatsBase
using Random

# include my codes -
include(joinpath(_PATH_TO_SRC, "CBOW.jl"));
include(joinpath(_PATH_TO_SRC, "SkipGram.jl"));
include(joinpath(_PATH_TO_SRC, "NegativeSampling.jl"));
