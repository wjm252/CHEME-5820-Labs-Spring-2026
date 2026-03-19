"""
    stochastic_attention_update(X::Matrix{Float32}, s::Vector{Float32}, β::Float64, η::Float64) -> Vector{Float32}

Perform a single stochastic attention update step (Algorithm 1 from L8a).
Given memory matrix `X`, current state `s`, inverse temperature `β`, and step size `η`,
returns the next state vector.
"""
function stochastic_attention_update(X::Matrix{Float32}, s::Vector{Float32},
    β::Float64, η::Float64)::Vector{Float32}

    N = length(s);
    z = transpose(X) * s;             # similarity vector (K × 1)
    p = softmax(β .* z);              # attention weights (K × 1)
    T_s = X * p;                      # attention pull (N × 1)
    ξ = randn(Float32, N);            # standard Gaussian noise
    σ = sqrt(2.0f0 * Float32(η) / Float32(β)); # noise amplitude

    return Float32((1 - η)) .* s .+ Float32(η) .* T_s .+ σ .* ξ;
end

"""
    deterministic_attention_update(X::Matrix{Float32}, s::Vector{Float32}, β::Float64, η::Float64) -> Vector{Float32}

Perform a single deterministic attention update step (no noise term).
"""
function deterministic_attention_update(X::Matrix{Float32}, s::Vector{Float32},
    β::Float64, η::Float64)::Vector{Float32}

    z = transpose(X) * s;             # similarity vector
    p = softmax(β .* z);              # attention weights
    T_s = X * p;                      # attention pull

    return Float32((1 - η)) .* s .+ Float32(η) .* T_s;
end

"""
    run_stochastic_attention(X, s₀, β, η, T; seed) -> Vector{Float32}

Run `T` steps of the stochastic attention update and return the final state.
"""
function run_stochastic_attention(X::Matrix{Float32}, s₀::Vector{Float32},
    β::Float64, η::Float64, T::Int; seed::Union{Int,Nothing} = nothing)::Vector{Float32}

    if seed !== nothing
        Random.seed!(seed);
    end

    s = copy(s₀);
    for _ in 1:T
        s = stochastic_attention_update(X, s, β, η);
    end
    return s;
end

"""
    run_deterministic_attention(X, s₀, β, η, T) -> Vector{Float32}

Run `T` steps of the deterministic attention update and return the final state.
"""
function run_deterministic_attention(X::Matrix{Float32}, s₀::Vector{Float32},
    β::Float64, η::Float64, T::Int)::Vector{Float32}

    s = copy(s₀);
    for _ in 1:T
        s = deterministic_attention_update(X, s, β, η);
    end
    return s;
end

"""
    run_stochastic_attention_averaged(X, s₀, β, η, T_burn, T_sample; seed) -> Vector{Float64}

Run `T_burn` steps of the stochastic attention update (burn-in, discarded), then run
`T_sample` additional steps and return the time-averaged attention weight vector
p̄ = (1/T_sample) Σ softmax(β X⊤ sᵗ). Averaging over the stationary distribution
reduces variance and gives a stable basis for classification.
"""
function run_stochastic_attention_averaged(X::Matrix{Float32}, s₀::Vector{Float32},
    β::Float64, η::Float64, T_burn::Int, T_sample::Int;
    seed::Union{Int,Nothing} = nothing)::Vector{Float64}

    if seed !== nothing
        Random.seed!(seed);
    end

    # burn-in: run to stationarity, discard states
    s = copy(s₀);
    for _ in 1:T_burn
        s = stochastic_attention_update(X, s, β, η);
    end

    # sampling: accumulate time-averaged attention weights
    p_accum = zeros(Float64, size(X, 2));
    for _ in 1:T_sample
        s = stochastic_attention_update(X, s, β, η);
        p_accum .+= softmax(β .* (transpose(X) * s));
    end

    return p_accum ./ T_sample;
end

"""
    run_stochastic_attention_trajectory(X, s₀, β, η, T; seed) -> Matrix{Float32}

Run `T` steps of the stochastic attention update and return the full trajectory.
Returns an `N × (T+1)` matrix where column `t` is the state at step `t-1` (column 1 is the initial state).
"""
function run_stochastic_attention_trajectory(X::Matrix{Float32}, s₀::Vector{Float32},
    β::Float64, η::Float64, T::Int; seed::Union{Int,Nothing} = nothing)::Matrix{Float32}

    if seed !== nothing
        Random.seed!(seed);
    end

    N = length(s₀);
    trajectory = Array{Float32,2}(undef, N, T + 1);
    trajectory[:, 1] = s₀;

    s = copy(s₀);
    for t in 1:T
        s = stochastic_attention_update(X, s, β, η);
        trajectory[:, t + 1] = s;
    end
    return trajectory;
end
