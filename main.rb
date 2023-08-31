$VERBOSE = nil

require 'csv'
require 'liblinear'

x_data = []
y_data = []

# Load data from CSV file into two arrays - one for independent variables X and one for the dependent variable Y
CSV.foreach("diabetes.csv", :headers => true) do |row|
  # Parse the row data
  Pregnancies = row["Pregnancies"].to_i
  Glucose = row["Glucose"].to_i
  BloodPressure = row["BloodPressure"].to_i
  SkinThickness = row["SkinThickness"].to_i
  Insulin = row["Insulin"].to_i
  BMI = row["BMI"].to_f
  DiabetesPedigreeFunction = row["DiabetesPedigreeFunction"].to_f
  Age = row["Age"].to_i
  
  # Prepare the feature vector
  x_data.push([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
  
  # Prepare the target variable
  y_data.push(row["Outcome"].to_i)
end

# Size of dataset
puts "Size of dataset: #{x_data.size}"

# Divide data into a training set and test set
test_size_percentage = 30.0
test_set_size = (x_data.size * (test_size_percentage / 100.0)).to_i 

len = y_data.size - test_set_size

# Size of training set
puts "Size of training data: #{len}"

# Size of test data
puts "Size of test data: #{test_set_size}"

test_x_data = x_data[len...y_data.size]
test_y_data = y_data[len...y_data.size]

training_x_data = x_data[0...len]
training_y_data = y_data[0...len]

# Setup model and train using training data
model = Liblinear.train(
  { solver_type: Liblinear::L2R_LR },   # Solver type: L2R_LR - L2-regularized logistic regression
  training_y_data,                      # Training data classification
  training_x_data,                      # Training data independent variables
  100                                   # Bias
)

# Example predictions for multiple instances
instances = [
  [6,148,72,35,0,33.6,0.627,50],
  [1,85,66,29,0,26.6,0.351,31],
  [8,183,64,0,0,23.3,0.672,32]
]

# Initialize arrays to store the probabilities
prediction_probs = []
probabilities = Array.new(instances.size) { Array.new(2, 0.0) }

# Make predictions for multiple instances
instances.each_with_index do |instance, index|
  prediction = Liblinear.predict(model, instance)
  probs = Liblinear.predict_probabilities(model, instance).sort
  prediction_probs << prediction
  probabilities[index] = probs
end

# Calculate the average probabilities
avg_probs = probabilities.transpose.map { |probs| probs.sum / probs.size }

# Print the average probabilities
avg_probs.each_with_index do |prob, index|
  puts "Average probabilities for class #{index}:"
  puts "#{(prob * 100).round(2)}%"
  puts "\n"
end

predicted = []
test_x_data.each do |params|
  predicted.push(Liblinear.predict(model, params))
end

correct = predicted.each_with_index.count { |e, i| e == test_y_data[i] }

accuracy = (correct.to_f / test_set_size) * 100.0
puts "Accuracy: #{accuracy.round(2)}%"
puts "Test set of size #{test_size_percentage}%"
