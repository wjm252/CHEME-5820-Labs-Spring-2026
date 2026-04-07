"""
    growth_rate(S_glc::Float64, S_gln::Float64, Lac::Float64, Amm::Float64,
        p::MyFedBatchCHOParameters) -> Float64

Compute the specific growth rate using Monod kinetics with dual by-product inhibition.

### Arguments
- `S_glc::Float64`: glucose concentration (mM).
- `S_gln::Float64`: glutamine concentration (mM).
- `Lac::Float64`: lactate concentration (mM).
- `Amm::Float64`: ammonia concentration (mM).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific growth rate (1/h).
"""
function growth_rate(S_glc::Float64, S_gln::Float64, Lac::Float64, Amm::Float64,
    p::MyFedBatchCHOParameters)::Float64

    # Monod terms for substrates -
    monod_glc = S_glc / (p.K_glc + S_glc);
    monod_gln = S_gln / (p.K_gln + S_gln);

    # inhibition terms for by-products -
    inhib_lac = p.K_I_lac / (p.K_I_lac + Lac);
    inhib_amm = p.K_I_amm / (p.K_I_amm + Amm);

    return p.mu_max * monod_glc * monod_gln * inhib_lac * inhib_amm;
end

"""
    product_formation_rate(p::MyFedBatchCHOParameters) -> Float64

Return the specific antibody productivity.

### Arguments
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific antibody productivity (mg/gDW/h).
"""
function product_formation_rate(p::MyFedBatchCHOParameters)::Float64
    return p.q_P;
end

"""
    substrate_uptake_glucose(mu::Float64, q_P::Float64,
        p::MyFedBatchCHOParameters) -> Float64

Compute the specific glucose uptake rate from growth and product formation demands.

### Arguments
- `mu::Float64`: specific growth rate (1/h).
- `q_P::Float64`: specific antibody productivity (mg/gDW/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific glucose uptake rate (mmol/gDW/h).
"""
function substrate_uptake_glucose(mu::Float64, q_P::Float64,
    p::MyFedBatchCHOParameters)::Float64

    return mu / p.Y_X_glc + q_P / p.Y_P_glc;
end

"""
    substrate_uptake_glutamine(mu::Float64, q_P::Float64,
        p::MyFedBatchCHOParameters) -> Float64

Compute the specific glutamine uptake rate from growth and product formation demands.

### Arguments
- `mu::Float64`: specific growth rate (1/h).
- `q_P::Float64`: specific antibody productivity (mg/gDW/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific glutamine uptake rate (mmol/gDW/h).
"""
function substrate_uptake_glutamine(mu::Float64, q_P::Float64,
    p::MyFedBatchCHOParameters)::Float64

    return mu / p.Y_X_gln + q_P / p.Y_P_gln;
end

"""
    byproduct_formation_lactate(q_glc::Float64,
        p::MyFedBatchCHOParameters) -> Float64

Compute the specific lactate formation rate from glucose consumption.

### Arguments
- `q_glc::Float64`: specific glucose uptake rate (mmol/gDW/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific lactate formation rate (mmol/gDW/h).
"""
function byproduct_formation_lactate(q_glc::Float64,
    p::MyFedBatchCHOParameters)::Float64

    return p.Y_lac_glc * q_glc;
end

"""
    byproduct_formation_ammonia(q_gln::Float64,
        p::MyFedBatchCHOParameters) -> Float64

Compute the specific ammonia formation rate from glutamine consumption.

### Arguments
- `q_gln::Float64`: specific glutamine uptake rate (mmol/gDW/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific ammonia formation rate (mmol/gDW/h).
"""
function byproduct_formation_ammonia(q_gln::Float64,
    p::MyFedBatchCHOParameters)::Float64

    return p.Y_amm_gln * q_gln;
end
