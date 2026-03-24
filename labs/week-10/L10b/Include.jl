# setup paths -
const _ROOT = pwd();
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# check: do we have a data directory?
!isdir(_PATH_TO_DATA) && mkpath(_PATH_TO_DATA);

# check do we have a Manifest.toml file?
using Pkg;
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using Statistics
using LinearAlgebra
using Random
using Downloads
using DataFrames
using CSV
using PrettyTables
using Flux
using NNlib
using OneHotArrays
using Plots
using JLD2
using Colors

# load local source files -
include(joinpath(_PATH_TO_SRC, "GloVeUtils.jl"));
include(joinpath(_PATH_TO_SRC, "Reviews.jl"));
include(joinpath(_PATH_TO_SRC, "Embeddings.jl"));
