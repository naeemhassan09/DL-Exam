using JLD2
using Statistics
using LinearAlgebra
using Flux

#############################
# NORMALIZATION FUNCTIONS
#############################
function apply_normalization(data::AbstractMatrix, norm_params::Dict)
    """
    Apply normalization using stored parameters
    Assumes data is [n_features × n_timesteps]
    """
    normalized_data = copy(Float32.(data))
    sensor_names = collect(keys(norm_params))  # dynamically use training sensor order

    for (i, sensor_name) in enumerate(sensor_names)
        if i <= size(data, 1) && haskey(norm_params, sensor_name)
            μ, σ = norm_params[sensor_name].μ, norm_params[sensor_name].σ
            normalized_data[i, :] = (normalized_data[i, :] .- μ) ./ σ
        end
    end
    return normalized_data
end

#############################
# WINDOWING FUNCTION
#############################
function create_windowed_sequences_test(data::AbstractMatrix, labels::AbstractVector, window_size::Int)
    n_features, n_timesteps = size(data)
    n_sequences = n_timesteps - window_size + 1
    if n_sequences <= 0
        error("Not enough data for window size $window_size")
    end

    X = zeros(Float32, n_features, window_size, n_sequences)
    y = zeros(Float32, n_sequences)

    for i in 1:n_sequences
        X[:, :, i] = data[:, i:(i+window_size-1)]
        y[i] = labels[i+window_size-1]   # consistent with training
    end
    return X, y
end

#############################
# MODEL RECONSTRUCTION
#############################
function create_model_from_params(n_features::Int, window_size::Int, architecture::String="mlp")
    if architecture == "mlp"
        flattened_size = n_features * window_size
        min_params_needed = flattened_size + 3  # input_to_hidden + hidden_bias + output

        if min_params_needed > 1000
            println("⚠️ Input too large, using temporal pooling.")
            return Chain(
                x -> mean(x, dims=2),
                Flux.flatten,
                Dense(n_features => 4, tanh),
                Dense(4 => 1, sigmoid)
            )
        else
            max_hidden = Int(floor((999 - 1) / (flattened_size + 2)))
            hidden_size = max(1, max_hidden)
            return Chain(
                Flux.flatten,
                Dense(flattened_size => hidden_size, tanh),
                Dense(hidden_size => 1, sigmoid)
            )
        end
    else
        error("Unsupported architecture: $architecture")
    end
end

#############################
# BALANCED ACCURACY FUNCTION
#############################
function bal_acc(args, trained_params, trained_st, test_x, test_y)
    try
        window_size = args[:window_size]
        normalization_params = args[:normalization_params]
        n_features = args[:n_features]
        architecture = get(args, :architecture, "mlp")

        println("🧪 Evaluating model with window size: $window_size")

        normalized_x = apply_normalization(test_x, normalization_params)
        X_test, y_test = create_windowed_sequences_test(normalized_x, test_y, window_size)

        println("   Test sequences: $(size(X_test, 3))")

        # Rebuild and load model
        model = create_model_from_params(n_features, window_size, architecture)
        Flux.loadparams!(model, trained_params)
        Flux.testmode!(model)

        # Predict
        y_pred = vec(model(X_test))
        y_pred_binary = y_pred .> 0.5

        # Confusion matrix
        tp = sum((y_pred_binary .== 1) .& (y_test .== 1))
        tn = sum((y_pred_binary .== 0) .& (y_test .== 0))
        fp = sum((y_pred_binary .== 1) .& (y_test .== 0))
        fn = sum((y_pred_binary .== 0) .& (y_test .== 1))

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        balanced_accuracy = (sensitivity + specificity) / 2.0

        println("   Sensitivity: $(round(sensitivity, digits=4))")
        println("   Specificity: $(round(specificity, digits=4))")
        println("   Balanced Accuracy: $(round(balanced_accuracy, digits=4))")

        return balanced_accuracy
    catch e
        println("❌ Error in bal_acc: $e")
        return 0.0
    end
end

#############################
# EVALUATION FUNCTIONS
#############################
function evaluate_model(model_path::String, window_size::Int, test_x, test_y)
    if !isfile(model_path)
        error("Model file not found: $model_path")
    end

    loaded_data = load(model_path)
    trained_params = loaded_data["trained_params"]
    trained_st     = loaded_data["trained_st"]
    model_data     = loaded_data["model_data"]

    args = Dict(
        :window_size => window_size,
        :normalization_params => model_data["normalization_params"],
        :n_features => model_data["n_features"],
        :architecture => model_data["architecture"]
    )

    return bal_acc(args, trained_params, trained_st, test_x, test_y)
end

function evaluate_all_models(test_x, test_y)
    window_sizes = [30, 90, 270]
    results = Dict()

    println("🎯 EVALUATING ALL MODELS")
    println("="^40)

    for ws in window_sizes
        model_path = "model_window_$(ws).jld2"
        if isfile(model_path)
            println("\n📊 Window Size $ws:")
            try
                acc = evaluate_model(model_path, ws, test_x, test_y)
                results[ws] = acc
                println("   ✅ Balanced Accuracy: $(round(acc, digits=4))")
            catch e
                println("   ❌ Evaluation failed: $e")
                results[ws] = 0.0
            end
        else
            println("⚠️ Model file not found: $model_path")
            results[ws] = 0.0
        end
    end

    println("\n📋 EVALUATION SUMMARY")
    best_window = argmax(values(results))
    println("🏆 Best Model: Window $(collect(keys(results))[best_window]) " *
            "(Accuracy: $(round(collect(values(results))[best_window], digits=4)))")

    return results
end

#############################
# LECTURER INTERFACE
#############################
if abspath(PROGRAM_FILE) == @__FILE__
    println("📝 test.jl loaded — ready for lecturer evaluation.")
    println("Use: results = evaluate_all_models(test_x, test_y)")
end