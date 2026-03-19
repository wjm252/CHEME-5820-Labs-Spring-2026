function convertcreditscore(s::Any)::Int64
    
    score = nothing;
    if (s == "Poor")
        score = 1;
    elseif (s == "Standard")
        score = 2;
    elseif (s == "Good")
        score = 3;
    end
    return score;
end

function convertcreditmix(s::Any)::Int64
    
    score = nothing;
    if (s == "Bad")
        score = -1;
    elseif (s == "Standard")
        score = 0;
    elseif (s == "Good")
        score = 1;
    end
    return score;
end

function convertcreditbehavior(s::Any)::Int64
    
    score = nothing;
    if (s == "Low_spent_Small_value_payments")
        score = 1;
    elseif (s == "Low_spent_Medium_value_payments")
        score = 2;
    elseif (s == "Low_spent_Large_value_payments")
        score = 3;
    elseif (s == "High_spent_Small_value_payments")
        score = 4;
    elseif (s == "High_spent_Medium_value_payments")
        score = 5;
    elseif (s == "High_spent_Large_value_payments")
        score = 6;
    end
    return score;
end