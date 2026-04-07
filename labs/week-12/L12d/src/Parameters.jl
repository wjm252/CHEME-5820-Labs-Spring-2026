"""
    build_default_parameters(; F_max, Glc_min, Glc_max, kwargs...) -> MyFedBatchCHOParameters

Construct a `MyFedBatchCHOParameters` instance with default mid-range kinetic values
from the Hockin-Mann CHO fed-batch model literature. The feed policy parameters
(`F_max`, `Glc_min`, `Glc_max`) are required keyword arguments because they serve
as conditioning inputs to the LSTM.

### Arguments (keyword)
- `F_max::Float64`: maximum feed rate (L/h). Required.
- `Glc_min::Float64`: glucose threshold to turn feed ON (mM). Required.
- `Glc_max::Float64`: glucose threshold to turn feed OFF (mM). Required.
- All other keyword arguments override default kinetic parameter values.

### Returns
- `MyFedBatchCHOParameters`: initialized parameter struct with `feed_on = 0.0`.
"""
function build_default_parameters(; F_max::Float64, Glc_min::Float64, Glc_max::Float64,
    mu_max::Float64 = 0.035,        # mid-range of 0.025-0.050 1/h
    K_glc::Float64 = 1.5,           # mid-range of 1.0-2.25 mM
    K_gln::Float64 = 0.12,          # mid-range of 0.047-0.23 mM
    K_I_lac::Float64 = 47.0,        # mid-range of 43-52 mM
    K_I_amm::Float64 = 8.0,         # mid-range of 6.5-9.5 mM
    k_d::Float64 = 0.005,           # mid-range of 0.003-0.007 1/h
    q_P::Float64 = 0.015,           # specific antibody productivity (mg/gDW/h)
    Y_X_glc::Float64 = 0.070,       # biomass yield on glucose (gDW/mmol)
    Y_X_gln::Float64 = 0.210,       # biomass yield on glutamine (gDW/mmol)
    Y_P_glc::Float64 = 0.05,        # product yield on glucose (mg/mmol)
    Y_P_gln::Float64 = 0.10,        # product yield on glutamine (mg/mmol)
    Y_lac_glc::Float64 = 1.2,       # lactate yield on glucose (mmol/mmol, mid-range 0.7-1.6)
    Y_amm_gln::Float64 = 0.70,      # ammonia yield on glutamine (mmol/mmol, mid-range 0.67-0.74)
    S_glc_f::Float64 = 500.0,       # feed glucose concentration (mM)
    S_gln_f::Float64 = 50.0,        # feed glutamine concentration (mM)
    )::MyFedBatchCHOParameters

    return MyFedBatchCHOParameters(
        mu_max, K_glc, K_gln, K_I_lac, K_I_amm, k_d,
        q_P,
        Y_X_glc, Y_X_gln, Y_P_glc, Y_P_gln, Y_lac_glc, Y_amm_gln,
        S_glc_f, S_gln_f,
        F_max, Glc_min, Glc_max,
        0.0  # feed starts OFF
    );
end
