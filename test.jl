
using JLD2
using Statistics
using LinearAlgebra
using Flux

#############################
# NORMALIZATION FUNCTIONS
#############################
function apply_normalization(data::AbstractMatrix, norm_params::Dict, sensor_names::Vector{String})
    """
    Apply normalization using stored parameters
    """
    normalized_data = copy(data)
    
    for (i, sensor_name) in enumerate(sensor_names)
        if haskey(norm_params, sensor_name)
            params = norm_params[sensor_name]
            μ = params.μ
            σ = params.σ
            normalized_data[i, :] = (normalized_data[i, :] .- μ) ./ σ
        end
    end
    
    return normalized_data
end

function create_windowed_sequences_test(data::AbstractMatrix, labels::AbstractVector, window_size::Int)
    """
    Create windowed sequences for testing (same as training)
    """
    n_features, n_timesteps = size(data)
    n_sequences = n_timesteps - window_size + 1
    
    if n_sequences <= 0
        error("Not enough data for window size $window_size")
    end
    
    X = zeros(Float32, n_features, window_size, n_sequences)
    y = zeros(Float32, n_sequences)
    
    for i in 1:n_sequences
        X[:, :, i] = data[:, i:(i+window_size-1)]
        # Use label from last timestep (same as training)
        y[i] = labels[i+window_size-1]
    end
    
    return X, y
end

#############################
# BALANCED ACCURACY FUNCTION
#############################
function bal_acc(args, trained_params, trained_st, test_x, test_y)
    """
    Main function to compute balanced accuracy
    Required interface for lecturer's evaluation
    
    Args:
        args: Dictionary containing model configuration
        trained_params: Model parameters (from JLD2)
        trained_st: Model state (from JLD2, may be nothing)
        test_x: Test input data [n_features × n_timesteps]
        test_y: Test labels [n_timesteps]
    
    Returns:
        balanced_accuracy: Float value between 0 and 1
    """
    
    try
        # Extract configuration
        window_size = args[:window_size]
        model_data = args[:model_data]
        model = args[:model]  # Use saved full model
        
        println("🧪 Evaluating model with window size: $window_size")
        
        # Get sensor names for normalization
        sensor_names = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", 
                       "Pressure", "Temperature", "Thermocouple", "Voltage", 
                       "Volume Flow RateRMS"]
        
        # Apply normalization using stored parameters
        normalized_x = apply_normalization(test_x, model_data["normalization_params"], sensor_names)
        
        # Create windowed sequences with labels
        X_test, y_test = create_windowed_sequences_test(normalized_x, test_y, window_size)
        
        println("   Test sequences: $(size(X_test, 3))")
        println("   Test labels: $(length(y_test))")
        
        # Ensure model is in evaluation mode
        Flux.testmode!(model)
        
        # Make predictions
        y_pred = vec(model(X_test))
        
        # Convert predictions to binary
        y_pred_binary = y_pred .> 0.5
        
        # Calculate balanced accuracy
        tp = sum((y_pred_binary .== 1) .& (y_test .== 1))
        tn = sum((y_pred_binary .== 0) .& (y_test .== 0))
        fp = sum((y_pred_binary .== 1) .& (y_test .== 0))
        fn = sum((y_pred_binary .== 0) .& (y_test .== 1))
        
        # Handle edge cases
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
# MODEL EVALUATION FUNCTIONS
#############################
function evaluate_model(model_path::String, window_size::Int, test_x, test_y)
    """
    Load and evaluate a specific model
    
    Args:
        model_path: Path to JLD2 model file
        window_size: Window size for the model
        test_x: Test input data
        test_y: Test labels
    
    Returns:
        balanced_accuracy: Performance metric
    """
    
    if !isfile(model_path)
        error("Model file not found: $model_path")
    end
    
    # Load model from JLD2 file
    loaded_data = load(model_path)
    trained_params = loaded_data["trained_params"]
    trained_st = loaded_data["trained_st"]
    model_data = loaded_data["model_data"]
    model = loaded_data["model"]  # Load full model
    
    # Prepare arguments
    args = Dict(
        :window_size => window_size,
        :model_data => model_data,
        :model => model
    )
    
    # Compute balanced accuracy
    return bal_acc(args, trained_params, trained_st, test_x, test_y)
end

function evaluate_all_models(test_x, test_y)
    """
    Evaluate all trained models and return results
    
    Args:
        test_x: Test input data [n_features × n_timesteps]
        test_y: Test labels [n_timesteps]
    
    Returns:
        results: Dictionary with results for each window size
    """
    
    window_sizes = [30, 90, 270]
    results = Dict()
    
    println("🎯 EVALUATING ALL MODELS")
    println("="^40)
    
    for window_size in window_sizes
        model_path = "model_window_$(window_size).jld2"
        
        if isfile(model_path)
            println("\n📊 Window Size $window_size:")
            
            try
                accuracy = evaluate_model(model_path, window_size, test_x, test_y)
                results[window_size] = accuracy
                println("   ✅ Balanced Accuracy: $(round(accuracy, digits=4))")
            catch e
                println("   ❌ Evaluation failed: $e")
                results[window_size] = 0.0
            end
        else
            println("\n⚠️  Model file not found: $model_path")
            results[window_size] = 0.0
        end
    end
    
    # Summary
    println("\n" * "="^40)
    println("📋 EVALUATION SUMMARY")
    println("="^40)
    
    best_window = 0
    best_accuracy = 0.0
    
    for window_size in window_sizes
        acc = results[window_size]
        println("Window $window_size: $(round(acc, digits=4))")
        
        if acc > best_accuracy
            best_accuracy = acc
            best_window = window_size
        end
    end
    
    if best_window > 0
        println("\n🏆 Best Model: Window $best_window (Accuracy: $(round(best_accuracy, digits=4)))")
    end
    
    return results
end

#############################
# EXAMPLE USAGE
#############################

# Example function showing how the lecturer will call this
function lecturer_evaluation_example()
    """
    Example of how the lecturer will evaluate the models
    This function shows the expected interface
    """
    
    # The lecturer will provide test_x and test_y
    # test_x: Matrix of size [n_features × n_timesteps] 
    # test_y: Vector of size [n_timesteps] with binary labels (0 or 1)
    
    println("📝 EXAMPLE EVALUATION INTERFACE")
    println("="^50)
    
    # Load and evaluate individual models
    window_sizes = [30, 90, 270]
    
    for window_size in window_sizes
        model_path = "model_window_$(window_size).jld2"
        
        if isfile(model_path)
            println("\nEvaluating $model_path:")
            
            # This is how lecturer will call it:
            # bal_acc_result = evaluate_model(model_path, window_size, test_x, test_y)
            # println("Balanced Accuracy: $bal_acc_result")
            
            println("✅ Model file exists and ready for evaluation")
        else
            println("❌ Model file missing: $model_path")
        end
    end
end

# Run example if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    lecturer_evaluation_example()
end

#############################
# DIRECT INTERFACE FOR LECTURER
#############################

"""
LECTURER USAGE INSTRUCTIONS:

1. For individual model evaluation:
   ```julia
   using JLD2
   include("test.jl")
   
   # Load your test data as test_x [features × timesteps], test_y [timesteps]
   accuracy_30 = evaluate_model("model_window_30.jld2", 30, test_x, test_y)
   accuracy_90 = evaluate_model("model_window_90.jld2", 90, test_x, test_y)
   accuracy_270 = evaluate_model("model_window_270.jld2", 270, test_x, test_y)
   ```

2. For evaluating all models at once:
   ```julia
   results = evaluate_all_models(test_x, test_y)
   ```

3. The bal_acc function can also be called directly:
   ```julia
   # Load model
   @load "model_window_30.jld2" trained_params trained_st model_data
   
   # Prepare args
   args = Dict(:window_size => 30, :model_data => model_data)
   
   # Calculate balanced accuracy
   accuracy = bal_acc(args, trained_params, trained_st, test_x, test_y)
   ```

Expected input format:
- test_x: Matrix [n_features × n_timesteps] with sensor readings
- test_y: Vector [n_timesteps] with binary labels (0=normal, 1=anomaly)

All models automatically apply the same normalization used during training.
"""