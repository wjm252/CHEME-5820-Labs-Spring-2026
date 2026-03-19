# setup paths -
const _ROOT = pwd();
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# flag to check if the include file was called -
const _DID_INCLUDE_FILE_GET_CALLED = true;

# check do we have a Manifest.toml file?
using Pkg;
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages
using Statistics
using LinearAlgebra
using Distributions
using PrettyTables
using DataFrames
using CSV
using FileIO
using Flux
using NNlib
using OneHotArrays
using Random
using JLD2
using Test

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));