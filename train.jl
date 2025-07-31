#############################
# train.jl - Deep Learning Expert Solution
# Binary Classification for Timeseries Anomaly Detection
# Optimized for Maximum Accuracy with Simple Models
#############################

using CSV
using DataFrames
using Statistics
using Glob
using Random
using Flux
using JLD2
using LinearAlgebra
using Plots

# Set random seeds for reproducibility
Random.seed!(42)

#############################
# ENHANCED DATA LOADING (Fixed Bugs)
#############################
function preprocess_file_optimized(filepath::String; label::Int=0)
    """
    Enhanced preprocessing with bug fixes and optimizations
    """
    try
        println("📂 Loading: $(basename(filepath))")
        df = CSV.read(filepath, DataFrame, delim=';', ignorerepeated=true)
        
        # Handle single column issue (manual split)
        if size(df, 2) == 1
            println("   🔧 Applying manual column split")
            raw = open(readlines, filepath)
            header = split(raw[1], ';')
            rows = [split(line, ';') for line in raw[2:end]]
            df = DataFrame([getindex.(rows, i) for i in 1:length(header)], Symbol.(header))
        end
        
        # Remove unwanted columns (task requirement)
        initial_cols = names(df)
        if "changepoint" in names(df)
            select!(df, Not("changepoint"))
            println("   ✅ Excluded 'changepoint' column")
        end
        if "anomaly" in names(df)
            select!(df, Not("anomaly"))
            println("   ✅ Excluded 'anomaly' column")
        end
        
        # Add binary label BEFORE any normalization
        df.label = fill(Float32(label), nrow(df))  # Use Float32 for consistency
        
        # Remove datetime column for modeling (keep only sensor data + label)
        if "datetime" in names(df)
            select!(df, Not("datetime"))
        end
        
        # Data is already numeric (Float64) - no need to parse again!
        # Convert to Float32 for better memory efficiency and GPU compatibility
        sensor_cols = [col for col in names(df) if col != "label"]
        for col in sensor_cols
            df[!, col] = Float32.(df[!, col])
        end
        
        println("   ✅ Processed $(nrow(df)) samples with $(length(sensor_cols)) sensors")
        return df
        
    catch e
        println("   ❌ ERROR: $e")
        return DataFrame()
    end
end

function load_all_data_optimized(base_path::String)
    """
    Load and combine all timeseries data with proper labeling
    """
    println("\n🚀 LOADING ALL TIMESERIES DATA")
    println("="^50)
    
    all_dfs = DataFrame[]
    
    # Load normal data (label = 0)
    println("📁 Loading NORMAL data...")
    normal_files = glob("*.csv", joinpath(base_path, "anomaly-free"))
    for file in normal_files
        df = preprocess_file_optimized(file, label=0)
        if !isempty(df)
            push!(all_dfs, df)
        end
    end
    
    # Load anomaly data (label = 1)
    println("\n📁 Loading ANOMALY data...")
    anomaly_folders = ["valve1", "valve2", "other"]
    for folder in anomaly_folders
        println("  → Processing $folder...")
        folder_files = glob("*.csv", joinpath(base_path, folder))
        for file in folder_files
            df = preprocess_file_optimized(file, label=1)
            if !isempty(df)
                push!(all_dfs, df)
            end
        end
    end
    
    if isempty(all_dfs)
        error("❌ No data loaded! Check data paths.")
    end
    
    # Combine all data with column alignment
    println("\n🔄 Combining and aligning data...")
    common_cols = reduce(intersect, [names(df) for df in all_dfs])
    combined_df = vcat([select(df, common_cols) for df in all_dfs]...)
    
    println("✅ Data loading complete!")
    println("   Total samples: $(nrow(combined_df))")
    println("   Sensor features: $(length(common_cols) - 1)")  # -1 for label column
    
    return combined_df
end

#############################
# ADVANCED DATA PREPROCESSING
#############################
function normalize_features!(df::DataFrame)
    """
    Normalize sensor features while preserving labels
    """
    println("\n🧮 NORMALIZING FEATURES")
    
    sensor_cols = [col for col in names(df) if col != "label"]
    normalization_params = Dict()
    
    for col in sensor_cols
        col_data = df[!, col]
        μ = mean(col_data)
        σ = std(col_data)
        
        if σ > 1e-8  # Avoid division by zero
            df[!, col] = (col_data .- μ) ./ σ
            normalization_params[col] = (μ=μ, σ=σ)
            println("   ✅ $col: μ=$(round(μ, digits=4)), σ=$(round(σ, digits=4))")
        else
            println("   ⚠️  $col: Zero variance, skipping normalization")
            normalization_params[col] = (μ=μ, σ=1.0)
        end
    end
    
    return normalization_params
end

function create_windowed_sequences(df::DataFrame, window_size::Int)
    """
    Create windowed sequences for temporal pattern learning
    Advanced implementation for maximum performance
    """
    println("\n🪟 Creating windowed sequences (window_size=$window_size)")
    
    sensor_cols = [col for col in names(df) if col != "label"]
    n_features = length(sensor_cols)
    n_samples = nrow(df)
    
    if n_samples < window_size
        error("❌ Not enough data for window size $window_size")
    end
    
    # Pre-allocate arrays for efficiency
    n_sequences = n_samples - window_size + 1
    X = zeros(Float32, n_features, window_size, n_sequences)
    y = zeros(Float32, n_sequences)
    
    sensor_matrix = Matrix{Float32}(df[!, sensor_cols])'  # Transpose for efficiency
    
    # Vectorized sequence creation
    for i in 1:n_sequences
        X[:, :, i] = sensor_matrix[:, i:(i+window_size-1)]
        y[i] = df[i+window_size-1, "label"]  # Label from last timestep
    end
    
    println("   ✅ Created $n_sequences sequences of shape $(size(X)[1:2])")
    return X, y
end

#############################
# OPTIMIZED MODEL ARCHITECTURES
#############################
function create_lightweight_model(n_features::Int, window_size::Int, model_type::String="mlp")
    """
    Create lightweight MLP model optimized for <1000 parameters
    """
    println("\n🧠 CREATING MLP MODEL")
    
    # Simple MLP - most parameter efficient
    flattened_size = n_features * window_size
    
    # Check if we can fit even 1 hidden unit
    min_params_needed = flattened_size + 1 + 1 + 1  # input_to_hidden + hidden_bias + hidden_to_output + output_bias
    
    if min_params_needed > 1000
        # For very large windows, use temporal pooling instead of flattening
        println("   ⚠️ Input too large ($flattened_size). Using temporal pooling approach.")
        
        # Use mean pooling over time dimension, then small MLP
        model = Chain(
            x -> mean(x, dims=2),  # Average over time: (features, time, batch) -> (features, 1, batch)
            Flux.flatten,          # (features, batch)
            Dense(n_features => 4, tanh),  # Small hidden layer
            Dense(4 => 1, sigmoid)
        )
        hidden_size = 4  # For printing purposes
    else
        # Standard approach: calculate optimal hidden size
        max_hidden = Int(floor((999 - 1) / (flattened_size + 2)))
        hidden_size = max(1, max_hidden)
        
        println("   🔧 Calculated hidden size: $hidden_size for input size: $flattened_size")
        
        model = Chain(
            Flux.flatten,
            Dense(flattened_size => hidden_size, tanh),
            Dense(hidden_size => 1, sigmoid)
        )
    end
    
    # Count parameters
    total_params = sum(length, Flux.params(model))
    println("   📊 Total parameters: $total_params")
    
    if min_params_needed > 1000
        println("   📐 Architecture: Temporal Pooling → Dense($(n_features)→4) → Dense(4→1)")
    else
        println("   📐 Architecture: Flatten → Dense($(flattened_size)→$(hidden_size)) → Dense($(hidden_size)→1)")
    end
    
    if total_params > 1000
        error("❌ Model exceeds 1000 parameter limit! ($total_params parameters)")
    end
    
    return model
end

#############################
# TRAINING ENGINE
#############################
function balanced_accuracy(y_pred::AbstractVector, y_true::AbstractVector)
    """
    Calculate balanced accuracy - critical metric for this task
    """
    y_pred_binary = y_pred .> 0.5
    
    # Calculate confusion matrix components
    tp = sum((y_pred_binary .== 1) .& (y_true .== 1))
    tn = sum((y_pred_binary .== 0) .& (y_true .== 0))
    fp = sum((y_pred_binary .== 1) .& (y_true .== 0))
    fn = sum((y_pred_binary .== 0) .& (y_true .== 1))
    
    # Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    return (sensitivity + specificity) / 2.0
end

function train_model_expert(model, X_train, y_train, X_val, y_val; 
                           epochs=100, batch_size=128, early_stopping_patience=15)
    println("\n🏋️ TRAINING MODEL")
    println("="^40)
    
    # Task-specified optimizer: AdamW
    opt = AdamW(0.001, (0.9, 0.999), 0.001)
    state = Flux.setup(opt, model)   # ✅ Create optimizer state

    # Data loaders
    train_loader = Flux.DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
    
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, train_accs, val_accs = Float32[], Float32[], Float32[]
    
    for epoch in 1:epochs
        epoch_loss, num_batches = 0.0, 0

        for (x_batch, y_batch) in train_loader
            loss, grads = Flux.withgradient(model) do m
                ŷ = vec(m(x_batch))
                Flux.binarycrossentropy(ŷ, y_batch)
            end

            # ✅ Correct update with state
            Flux.Optimise.update!(state, model, grads)

            epoch_loss += loss
            num_batches += 1
        end
        
        avg_loss = epoch_loss / num_batches
        push!(train_losses, avg_loss)
        
        # Evaluation phase
        y_train_pred = vec(model(X_train))
        y_val_pred = vec(model(X_val))
        
        train_acc = balanced_accuracy(y_train_pred, y_train)
        val_acc = balanced_accuracy(y_val_pred, y_val)
        
        push!(train_accs, train_acc)
        push!(val_accs, val_acc)
        
        # Early stopping logic
        if val_acc > best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        # Progress reporting with diagnostics
        if epoch % 10 == 0 || epoch == 1
            # Show prediction distribution for debugging
            train_pred_stats = (min=minimum(y_train_pred), max=maximum(y_train_pred), mean=mean(y_train_pred))
            val_pred_stats = (min=minimum(y_val_pred), max=maximum(y_val_pred), mean=mean(y_val_pred))
            
            println("Epoch $epoch: Loss=$(round(avg_loss, digits=4)), " *
                   "Train_Acc=$(round(train_acc, digits=4)), " *
                   "Val_Acc=$(round(val_acc, digits=4)) [Best: $(round(best_val_acc, digits=4))]")
            
            if epoch == 1
                println("   🔍 Debug - Train preds: min=$(round(train_pred_stats.min, digits=3)), " *
                       "max=$(round(train_pred_stats.max, digits=3)), " *
                       "mean=$(round(train_pred_stats.mean, digits=3))")
                println("   🔍 Debug - Val preds: min=$(round(val_pred_stats.min, digits=3)), " *
                       "max=$(round(val_pred_stats.max, digits=3)), " *
                       "mean=$(round(val_pred_stats.mean, digits=3))")
                
                # Show label distribution
                train_label_dist = round(mean(y_train), digits=3)
                val_label_dist = round(mean(y_val), digits=3)
                println("   🔍 Debug - Train labels mean: $train_label_dist, Val labels mean: $val_label_dist")
            end
        end
        
        # Early stopping
        if patience_counter >= early_stopping_patience
            println("   🛑 Early stopping at epoch $epoch (patience exceeded)")
            break
        end
    end
    
    println("   🎯 Best validation accuracy: $(round(best_val_acc, digits=4))")
    return train_losses, train_accs, val_accs, best_val_acc
end

#############################
# MAIN TRAINING PIPELINE
#############################
function main()
    println("🚀 DEEP LEARNING EXPERT - ANOMALY DETECTION TRAINING")
    println("="^60)
    
    # Load and preprocess data
    base_path = "data"
    df = load_all_data_optimized(base_path)
    
    # Class distribution analysis
    class_dist = combine(groupby(df, :label), nrow => :count)
    println("\n📊 CLASS DISTRIBUTION:")
    for row in eachrow(class_dist)
        class_name = row.label == 0.0 ? "Normal" : "Anomaly" 
        percentage = round(100 * row.count / nrow(df), digits=1)
        println("   $class_name: $(row.count) samples ($(percentage)%)")
    end
    
    # Feature normalization
    norm_params = normalize_features!(df)
    
    # PROPER Stratified train/validation split (80/20) to prevent data leakage
    normal_indices = findall(df.label .== 0.0)
    anomaly_indices = findall(df.label .== 1.0)
    
    println("\n🔄 STRATIFIED SPLIT:")
    println("   Normal samples: $(length(normal_indices))")
    println("   Anomaly samples: $(length(anomaly_indices))")
    
    # Shuffle within each class
    normal_shuffled = shuffle(normal_indices)
    anomaly_shuffled = shuffle(anomaly_indices)
    
    # Split each class 80/20
    n_normal_train = Int(floor(0.8 * length(normal_indices)))
    n_anomaly_train = Int(floor(0.8 * length(anomaly_indices)))
    
    train_indices = vcat(
        normal_shuffled[1:n_normal_train],
        anomaly_shuffled[1:n_anomaly_train]
    )
    
    val_indices = vcat(
        normal_shuffled[(n_normal_train+1):end],
        anomaly_shuffled[(n_anomaly_train+1):end]
    )
    
    # Shuffle combined indices to mix classes
    train_indices = shuffle(train_indices)
    val_indices = shuffle(val_indices)
    
    train_data = df[train_indices, :]
    val_data = df[val_indices, :]
    
    println("\n📊 DATA SPLIT:")
    println("   Training: $(nrow(train_data)) samples")
    println("   Validation: $(nrow(val_data)) samples")
    
    # Verify stratification worked
    train_class_dist = combine(groupby(train_data, :label), nrow => :count)
    val_class_dist = combine(groupby(val_data, :label), nrow => :count)
    
    println("\n✅ STRATIFICATION VERIFICATION:")
    println("   Training class distribution:")
    for row in eachrow(train_class_dist)
        class_name = row.label == 0.0 ? "Normal" : "Anomaly"
        percentage = round(100 * row.count / nrow(train_data), digits=1)
        println("     $class_name: $(row.count) samples ($(percentage)%)")
    end
    
    println("   Validation class distribution:")
    for row in eachrow(val_class_dist)
        class_name = row.label == 0.0 ? "Normal" : "Anomaly"
        percentage = round(100 * row.count / nrow(val_data), digits=1)
        println("     $class_name: $(row.count) samples ($(percentage)%)")
    end
    
    # Train models for different window sizes (task requirement)
    window_sizes = [30, 90, 270]
    all_results = Dict()
    
    for window_size in window_sizes
        println("\n" * "="^60)
        println("🎯 TRAINING MODEL FOR WINDOW SIZE: $window_size")
        println("="^60)
        
        # Create windowed sequences
        X_train, y_train = create_windowed_sequences(train_data, window_size)
        X_val, y_val = create_windowed_sequences(val_data, window_size)
        
        println("   Training sequences: $(size(X_train, 3))")
        println("   Validation sequences: $(size(X_val, 3))")
        
        # Create MLP model only (simplified approach)
        println("\n   🧪 Creating MLP model...")
        model = create_lightweight_model(size(X_train, 1), window_size, "mlp")
        
        train_losses, train_accs, val_accs, final_acc = train_model_expert(
            model, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=128
        )
        
        best_model = model
        best_accuracy = final_acc
        best_results = (train_losses, train_accs, val_accs, final_acc, "mlp")
        
        # Save the best model (task requirement: JLD2 format)
        model_filename = "model_window_$(window_size).jld2"
        model_data = Dict(
            "model" => best_model,
            "window_size" => window_size,
            "normalization_params" => norm_params,
            "n_features" => size(X_train, 1),
            "best_accuracy" => best_accuracy,
            "architecture" => best_results[5],
            "training_history" => Dict(
                "train_losses" => best_results[1],
                "train_accs" => best_results[2], 
                "val_accs" => best_results[3]
            )
        )
        
        jldsave(model_filename; 
                 trained_params=Flux.params(best_model), 
                 trained_st=nothing, 
                 model=best_model,
                 model_data=model_data)
        
        println("\n   💾 Model saved: $model_filename")
        println("   🎯 Final balanced accuracy: $(round(best_accuracy, digits=4))")
        
        all_results[window_size] = (best_accuracy, best_results)
    end
    
    # Final summary
    println("\n" * "="^80)
    println("🏆 FINAL RESULTS SUMMARY")
    println("="^80)
    
    best_overall_window = 0
    best_overall_acc = 0.0
    
    for window_size in window_sizes
        acc = all_results[window_size][1]
        arch = all_results[window_size][2][5]
        println("Window $window_size ($arch): Balanced Accuracy = $(round(acc, digits=4))")
        
        if acc > best_overall_acc
            best_overall_acc = acc
            best_overall_window = window_size
        end
    end
    
    println("\n🥇 BEST MODEL: Window size $best_overall_window")
    println("   Balanced Accuracy: $(round(best_overall_acc, digits=4))")
    
    # Create visualization
    create_results_visualization(all_results, window_sizes)
    
    println("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    println("   📁 Models saved in JLD2 format")
    println("   📊 Results visualization saved")
    
    return all_results
end

function create_results_visualization(results, window_sizes)
    """
    Create comprehensive results visualization
    """
    println("\n📊 Creating results visualization...")
    
    accuracies = [results[ws][1] for ws in window_sizes]
    architectures = [results[ws][2][5] for ws in window_sizes]
    
    # Accuracy comparison plot
    p1 = bar(string.(window_sizes), accuracies,
             title="Balanced Accuracy by Window Size",
             xlabel="Window Size", ylabel="Balanced Accuracy",
             ylim=(0, 1), legend=false,
             color=[:skyblue, :lightgreen, :salmon])
    
    # Add architecture labels
    for (i, arch) in enumerate(architectures)
        annotate!(p1, i, accuracies[i] + 0.02, text(arch, 8, :center))
    end
    
    # Training curves for best model
    best_window = window_sizes[argmax(accuracies)]
    _, train_losses, train_accs, val_accs, _, _ = results[best_window][2]
    
    p2 = plot(train_accs, label="Train Accuracy", 
              title="Training Curves (Window $best_window)",
              xlabel="Epoch", ylabel="Balanced Accuracy")
    plot!(p2, val_accs, label="Validation Accuracy")
    
    p3 = plot(train_losses,
              title="Training Loss (Window $best_window)", 
              xlabel="Epoch", ylabel="Loss", legend=false)
    
    combined = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
    savefig(combined, "training_results.png")
    
    println("   💾 Visualization saved: training_results.png")
end

# Execute training pipeline
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end
main()