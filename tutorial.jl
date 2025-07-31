ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
using MLDatasets

# Helper function to safely load datasets
function test_dataset(name, symbol)
    if hasproperty(MLDatasets, symbol)
        dataset = getproperty(MLDatasets, symbol)
        try
            train_x, train_y = dataset(:train)[:]
            test_x, test_y   = dataset(:test)[:]
            println("✅ $name loaded successfully")
            println("   Train: ", size(train_x), " | ", size(train_y))
            println("   Test : ", size(test_x), " | ", size(test_y))
        catch e
            println("❌ $name failed to load: ", e)
        end
    else
        println("⚠️ $name not available in this version of MLDatasets")
    end
    println("-"^50)
end

println("🔎 Checking available MLDatasets...\n")

# Candidate datasets (will check dynamically if available)
datasets = [
    ("MNIST", :MNIST),
    ("FashionMNIST", :FashionMNIST),
    ("CIFAR10", :CIFAR10),
    ("CIFAR100", :CIFAR100),
    ("SVHN2", :SVHN2),
    ("COIL20", :COIL20),
    ("NotMNIST", :NotMNIST),
    ("KuzushijiMNIST", :KuzushijiMNIST)
]

for (name, sym) in datasets
    test_dataset(name, sym)
end