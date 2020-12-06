# Import Packages
using Pkg  # Package to install new packages

# Install packages 
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Plots")
Pkg.add("Lathe")
Pkg.add("GLM")
Pkg.add("StatsPlots")
Pkg.add("MLBase")
Pkg.add("RDatasets")

# Load the installed packages
using DataFrames
using CSV
using Plots
using Lathe
using GLM
using Statistics
using StatsPlots
using MLBase
using RDatasets
using Random

# Enable printing of 1000 columns
ENV["COLUMNS"] = 1000

Ocean_df = CSV.read("C:\\Users\\HP\\Downloads\\bottle.csv");

first(Ocean_df,5)

last(Ocean_df,5)

println(size(Ocean_df))

describe(Ocean_df)

# Check column names
(names(Ocean_df))

calcofi_subset = Ocean_df[:,[:T_degC ,:Salnty,:Depthm,:O2ml_L, :STheta, :O2Sat]] # :Oxy_µmol/Kg]]

names(calcofi_subset)

rename!(calcofi_subset,Symbol("T_degC")=>:Temprature_degC)
rename!(calcofi_subset,Symbol("Salnty")=>:Salinity)
rename!(calcofi_subset,Symbol("Depthm")=>:Depth_mtr)
rename!(calcofi_subset,Symbol("O2ml_L")=>:O2_ml_per_ltr)

describe(calcofi_subset)

dropmissing!(calcofi_subset,[:Temprature_degC,:Salinity,:Depth_mtr,:O2_ml_per_ltr, :STheta, :O2Sat])

calcofi_subset

boxplot(calcofi_subset.Temprature_degC , title = "Box Plot - Water Temprature", ylabel = "Water Temprature(deg_C)", legend = false)

# Outlier removal -above
first_percentile = percentile(calcofi_subset.Temprature_degC, 95)
iqr_value = iqr(calcofi_subset.Temprature_degC)
calcofi_subset = calcofi_subset[calcofi_subset.Temprature_degC .<  (first_percentile - 1.5*iqr_value),:];

# Outlier removal -below
first_percentile = percentile(calcofi_subset.Temprature_degC, 25)
iqr_value = iqr(calcofi_subset.Temprature_degC)
calcofi_subset = calcofi_subset[calcofi_subset.Temprature_degC .>  (first_percentile - 1.5*iqr_value),:];

# Density Plot
density(calcofi_subset.Temprature_degC , title = "Density Plot - Water Temprature  ", ylabel = "Frequency", xlabel = "Water Temprature", legend = false)

maximum(calcofi_subset.Temprature_degC)

minimum(calcofi_subset.Temprature_degC)

median(calcofi_subset.Temprature_degC)

describe(calcofi_subset)

# Setting Data
Target = calcofi_subset[!, :Temprature_degC	];
Salinity = calcofi_subset[!, :Salinity];



# Correlation Analysis

println("Correlation of Water Salanity and Water Temprature is ", cor(calcofi_subset.Temprature_degC,calcofi_subset.Salinity), "\n\n")

# Scatter Plots
 Plots.pyplot()

P = scatter(Salinity,Target,labels=false,markerstrokewidth=0,markersize=3,alpha=0.6,)
xlabel!("Water Salinity")
ylabel!("Water Temprature")
@show P

# Train test split
using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(calcofi_subset,0.75)

function partitionTrainTest(calcofi_subset, at = 0.7)
    n = nrow(calcofi_subset)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    calcofi_subset[train_idx,:], calcofi_subset[test_idx,:]
end

# iris = dataset("datasets", "calcofi_subset")
 train,test = partitionTrainTest(calcofi_subset, 0.7)

fm = @formula(Temprature_degC ~ Salinity)
linearRegressor = lm(fm, train)

# R Square value of the model
r2(linearRegressor)


# Prediction
ypredicted_test = predict(linearRegressor, test)
ypredicted_train = predict(linearRegressor, train)


# Test Performance DataFrame (compute squared error)
performance_testdf = DataFrame(y_actual = test[!,:Temprature_degC], y_predicted = ypredicted_test)
performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error

# Train Performance DataFrame (compute squared error)
performance_traindf = DataFrame(y_actual = train[!,:Temprature_degC], y_predicted = ypredicted_train)
performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error ;

# MAPE function defination
function mape(performance_df)
    mape = mean(abs.(performance_df.error./performance_df.y_actual))
    return mape
end

# RMSE function defination
function rmse(performance_df)
    rmse = sqrt(mean(performance_df.error.*performance_df.error))
    return rmse
end

# Test Error
println("Mean Absolute test error: ",mean(abs.(performance_testdf.error)), "\n")
println("Mean Aboslute Percentage test error: ",mape(performance_testdf), "\n")
println("Root mean square test error: ",rmse(performance_testdf), "\n")
println("Mean square test error: ",mean(performance_testdf.error_sq), "\n")

# Train  Error
println("Mean train error: ",mean(abs.(performance_traindf.error)), "\n")
println("Mean Absolute Percentage train error: ",mape(performance_traindf), "\n")
println("Root mean square train error: ",rmse(performance_traindf), "\n")
println("Mean square train error: ",mean(performance_traindf.error_sq), "\n")

# Histogram of error to see if it's normally distributed  on test dataset
histogram(performance_testdf.error, bins = 50, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)

# Histogram of error to see if it's normally distributed  on train dataset
histogram(performance_traindf.error, bins = 50, title = "Training Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)

# Scatter plot of actual vs predicted values on test dataset
test_plot = scatter(performance_testdf[!,:y_actual],performance_testdf[!,:y_predicted], title = "Predicted value vs Actual value on Test Data", ylabel = "Predicted value", xlabel = "Actual value", legend = false)

# Scatter plot of actual vs predicted values on train dataset
train_plot = scatter(performance_traindf[!,:y_actual],performance_traindf[!,:y_predicted], title = "Predicted value vs Actual value on Train Data", ylabel = "Predicted value", xlabel = "Actual value",legend = false)

# Cross Validation function defination
function cross_validation(train,k, fm = @formula(Temprature_degC ~ Salinity))
    a = collect(Kfold(size(train)[1], k))
    for i in 1:k
        row = a[i]
        temp_train = train[row,:]
        temp_test = train[setdiff(1:end, row),:]
        linearRegressor = lm(fm, temp_train)
        performance_testdf = DataFrame(y_actual = temp_test[!,:Temprature_degC], y_predicted = predict(linearRegressor, temp_test))
        performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]

        println("Mean error for set $i is ",mean(abs.(performance_testdf.error)))
    end
end

cross_validation(train,10)

describe(calcofi_subset)

fm = @formula(Temprature_degC ~ Salinity + Depth_mtr + O2_ml_per_ltr + STheta + O2Sat)
linearRegressor = lm(fm, train)

# R Square value of the model
r2(linearRegressor)

# Prediction
ypredicted_test = predict(linearRegressor, test)
ypredicted_train = predict(linearRegressor, train)

# Test Performance DataFrame
performance_testdf = DataFrame(y_actual = test[!,:Temprature_degC], y_predicted = ypredicted_test)
performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error

# Train Performance DataFrame
performance_traindf = DataFrame(y_actual = train[!,:Temprature_degC], y_predicted = ypredicted_train)
performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error ;

# Test Error
println("Mean Absolute test error: ",mean(abs.(performance_testdf.error)), "\n")
println("Mean Aboslute Percentage test error: ",mape(performance_testdf), "\n")
println("Root mean square test error: ",rmse(performance_testdf), "\n")
println("Mean square test error: ",mean(performance_testdf.error_sq), "\n")

# Train  Error
println("Mean train error: ",mean(abs.(performance_traindf.error)), "\n")
println("Mean Aboslute Percentage train error: ",mape(performance_traindf), "\n")
println("Root mean square train error: ",rmse(performance_traindf), "\n")
println("Mean square train error: ",mean(performance_traindf.error_sq), "\n")

# Histogram of error to see if it's normally distributed  on test dataset
histogram(performance_testdf.error, bins = 150, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)

# Histogram of error to see if it's normally distributed  on train dataset
histogram(performance_traindf.error, bins = 150, title = "Training Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)

# Scatter plot of actual vs predicted values on test dataset
test_plot = scatter(performance_testdf[!,:y_actual],performance_testdf[!,:y_predicted], title = "Predicted value vs Actual value on Test Data", ylabel = "Predicted value", xlabel = "Actual value", legend = false)

# Scatter plot of actual vs predicted values on train dataset
train_plot = scatter(performance_traindf[!,:y_actual],performance_traindf[!,:y_predicted], title = "Predicted value vs Actual value on Train Data", ylabel = "Predicted value", xlabel = "Actual value",legend = false)

cross_validation(train,10, fm)
