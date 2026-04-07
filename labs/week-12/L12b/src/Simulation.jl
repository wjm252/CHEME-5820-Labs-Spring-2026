"""
    generate_hemophilia_dataset(fviii_percent_values::Vector{Float64};
        TF::Float64 = 5e-12, tspan::Tuple{Float64,Float64} = (0.0, 1200.0),
        saveat::Float64 = 5.0) -> Tuple

Generate a dataset of thrombin generation curves by simulating the Hockin-Mann 2002 coagulation
model at each Factor VIII level in `fviii_percent_values` (as percent of nominal).

### Arguments
- `fviii_percent_values::Vector{Float64}`: vector of FVIII levels in percent nominal (0 to 100).
- `TF::Float64`: tissue factor concentration in molar (default: 5 pM).
- `tspan::Tuple{Float64,Float64}`: simulation time span in seconds (default: 0 to 1200).
- `saveat::Float64`: time step for saving solution points in seconds (default: 5.0).

### Returns
- `Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}}`: tuple of (time_vector, thrombin_matrix, fviii_percent_values)
   where `thrombin_matrix` has rows as time points and columns as different FVIII levels.
   Thrombin values are in nanomolar (nM).
"""
function generate_hemophilia_dataset(fviii_percent_values::Vector{Float64};
    TF::Float64 = 5e-12, tspan::Tuple{Float64,Float64} = (0.0, 1200.0),
    saveat::Float64 = 5.0)::Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}}

    # how many curves do we need to generate? -
    n_curves = length(fviii_percent_values);

    # run the first simulation outside the loop so we can size the output matrix.
    # we need to know how many time points the ODE solver will return before we
    # can allocate the thrombin_matrix, and that depends on tspan and saveat -
    fviii_molar_first = HockinMannModel.percent_nominal_to_molar(fviii_percent_values[1], :VIII); # convert clinical % to molar
    u0_first = HockinMannModel.patient_initial_conditions(HockinMann2002; TF=TF, VIII=fviii_molar_first); # build initial state vector with custom FVIII level
    sol_first = HockinMannModel.simulate(HockinMann2002; u0=u0_first, tspan=tspan, saveat=saveat); # run the ODE solver
    time_vector = sol_first.t;                              # shared time grid for all curves
    n_timepoints = length(time_vector);                     # number of saved time points
    thrombin_matrix = zeros(Float64, n_timepoints, n_curves); # rows = time, cols = different FVIII levels

    # simulate the Hockin-Mann coagulation model once per FVIII level -
    for (j, fviii_pct) in enumerate(fviii_percent_values)

        # convert FVIII from clinical percent of nominal to molar concentration -
        fviii_molar = HockinMannModel.percent_nominal_to_molar(fviii_pct, :VIII);

        # build the initial condition vector with the requested FVIII level -
        u0 = HockinMannModel.patient_initial_conditions(HockinMann2002; TF=TF, VIII=fviii_molar);

        # integrate the ODE system over tspan, returning a solution sampled at `saveat` -
        sol = HockinMannModel.simulate(HockinMann2002; u0=u0, tspan=tspan, saveat=saveat);

        # extract the total thrombin trajectory (free + bound forms) and convert M to nM -
        # the model returns concentrations in molar; nM is the standard reporting unit for TGA -
        thrombin_matrix[:, j] = HockinMannModel.total_thrombin(HockinMann2002, sol) .* 1e9;
    end

    # return time grid, the curves, and the FVIII levels (echoed for convenience) -
    return (time_vector, thrombin_matrix, fviii_percent_values)
end

"""
    normalize_minmax(data::Matrix{Float64}) -> Tuple{Matrix{Float64}, Float64, Float64}

Apply min-max normalization to scale all values in `data` to the interval [0, 1].

### Arguments
- `data::Matrix{Float64}`: matrix of values to normalize.

### Returns
- `Tuple{Matrix{Float64}, Float64, Float64}`: tuple of (normalized_data, data_min, data_max).
"""
function normalize_minmax(data::Matrix{Float64})::Tuple{Matrix{Float64}, Float64, Float64}

    # find a single global min and max across the entire matrix -
    # using one pair of bounds for all curves is what makes this "global"
    # normalization: amplitude differences between curves are preserved -
    data_min = minimum(data);
    data_max = maximum(data);

    # affine map: x -> (x - min) / (max - min) sends [min, max] to [0, 1] -
    normalized = (data .- data_min) ./ (data_max - data_min);

    # return the bounds alongside the normalized data so callers can later
    # invert the transformation with denormalize_minmax(...) -
    return (normalized, data_min, data_max)
end

"""
    normalize_minmax_percurve(data::Matrix{Float64}) -> Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}

Apply min-max normalization independently to each column (curve) of `data`, scaling each
column to the interval [0, 1] using that column's own minimum and maximum.

Normalizing per-curve rather than globally ensures that:
- Test-curve statistics do not pollute the normalization applied to training data.
- The peak of every curve is mapped to 1.0, giving the peak region proportional weight
  in the MSE loss regardless of its absolute thrombin level.

### Arguments
- `data::Matrix{Float64}`: matrix of shape `(T, n_curves)` where each column is one curve.

### Returns
- `Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}`: tuple of
  (normalized_data, curve_mins, curve_maxs) where `curve_mins` and `curve_maxs` are
  vectors of length `n_curves` holding the per-column normalization bounds.
"""
function normalize_minmax_percurve(data::Matrix{Float64})::Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}

    # allocate output matrix and per-column bound vectors -
    # each column gets its own (min, max) pair, which is the difference between
    # this and the global normalize_minmax(...) above -
    n_curves = size(data, 2);
    normalized = similar(data);
    curve_mins = Vector{Float64}(undef, n_curves);
    curve_maxs = Vector{Float64}(undef, n_curves);

    # walk through one column (one curve) at a time -
    for j in 1:n_curves

        # find this curve's own min and max -
        col_min = minimum(data[:, j]);
        col_max = maximum(data[:, j]);

        # store the bounds so this curve can be denormalized later -
        curve_mins[j] = col_min;
        curve_maxs[j] = col_max;

        # rescale this curve to [0, 1] using its own bounds: every curve's peak
        # ends up at exactly 1.0, which destroys the amplitude signal but
        # gives the peak region equal weight in any per-curve loss -
        normalized[:, j] = (data[:, j] .- col_min) ./ (col_max - col_min);
    end

    return (normalized, curve_mins, curve_maxs)
end

"""
    denormalize_minmax(data::Vector{Float64}, data_min::Float64, data_max::Float64) -> Vector{Float64}

Reverse min-max normalization for a vector.

### Arguments
- `data::Vector{Float64}`: normalized vector with values in [0, 1].
- `data_min::Float64`: minimum value from the original data.
- `data_max::Float64`: maximum value from the original data.

### Returns
- `Vector{Float64}`: vector in original scale.
"""
function denormalize_minmax(data::Vector{Float64}, data_min::Float64, data_max::Float64)::Vector{Float64}

    # inverse of (x - min) / (max - min): scale by the original range, then shift back -
    # used to map model predictions in [0, 1] back to physical units (nM thrombin) -
    return data .* (data_max - data_min) .+ data_min
end
