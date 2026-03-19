function build_weighted_cooccurrence_matrix(sentences::Array{String,1}, vocabulary::Dict{String,Int64}; window_size::Int64=2)::Array{Float64,2}
    vocab_size = length(vocabulary)
    X = zeros(Float64, vocab_size, vocab_size)
    for sentence in sentences
        words = split(lowercase(sentence))
        n = length(words)
        for i in 1:n
            w_idx = get(vocabulary, words[i], get(vocabulary, "<unk>", 0))
            if w_idx == 0; continue; end
            for j in max(1,i-window_size):min(n,i+window_size)
                if j == i; continue; end
                ctx_idx = get(vocabulary, words[j], get(vocabulary, "<unk>", 0))
                if ctx_idx > 0
                    d = abs(i - j)
                    X[w_idx, ctx_idx] += 1.0 / d
                end
            end
        end
    end
    return X
end

function glove_weight(x::Float64; x_max::Float64=100.0, alpha::Float64=0.75)::Float64
    return min(1.0, (x / x_max)^alpha)
end

function train_glove(X::Array{Float64,2}, vocab_size::Int64; d::Int64=5, eta::Float64=0.05,
    num_epochs::Int64=500, x_max::Float64=100.0, alpha::Float64=0.75)
    W_word = randn(d, vocab_size) * 0.01
    W_ctx  = randn(d, vocab_size) * 0.01
    b_word = zeros(Float64, vocab_size)
    b_ctx  = zeros(Float64, vocab_size)
    loss_history = zeros(Float64, num_epochs)
    nonzero_pairs = [(i,j) for i in 1:vocab_size for j in 1:vocab_size if X[i,j] > 0]
    for epoch in 1:num_epochs
        total_loss = 0.0
        for (i, j) in nonzero_pairs
            xij = X[i, j]
            fw = glove_weight(xij; x_max=x_max, alpha=alpha)
            diff = dot(W_word[:,i], W_ctx[:,j]) + b_word[i] + b_ctx[j] - log(xij)
            loss = fw * diff^2
            total_loss += loss
            grad = 2.0 * fw * diff
            W_word[:,i] .-= eta .* grad .* W_ctx[:,j]
            W_ctx[:,j]  .-= eta .* grad .* W_word[:,i]
            b_word[i]   -= eta * grad
            b_ctx[j]    -= eta * grad
        end
        loss_history[epoch] = total_loss / length(nonzero_pairs)
    end
    return W_word, W_ctx, b_word, b_ctx, loss_history
end
