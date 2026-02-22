# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Module 2: Critical Thinking Assignment
# Option #2: Predicting Future Sales
#
# Now that the data has been scaled, you’re ready to code the neural network. You’ll code a neural network with Keras.
# To do this, you’ll complete the given Python script named create_model.py, shown below.
#
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import *
#
# training_data_df = pd.read_csv("sales_data_training_scaled.csv")
# X = training_data_df.drop('total_earnings', axis=1).values
# Y = training_data_df[['total_earnings']].values  # Define the model  model =
#
# Train the model
# model.fit(…)
#
# Load the test data
# Load the separate test data set
# test_data_df = pd.read_csv("sales_data_test_scaled.csv")
#
# X_test = test_data_df.drop('total_earnings', axis=1).values
# Y_test = test_data_df[['total_earnings']].values
#
# First, on line five, use the Python package, Pandas library, (visit the Python Pandas Tutorial,
# https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/,
# website for more information) to load the pre - scaled data from a CSV file. Each row of the dataset contains several
# features that describe each video game and then the total earnings value for that game. You want to split that data
# into two separate arrays: one with just the input features for each game and one with just the expected earnings.
#
# On line seven, to get just the input features, we grab all of the columns of the training data but drop the total
# earnings column. Then, on line eight, extract just the total earnings column as shown. Now, X contains all the input
# features for each game, and Y contains only the expected earnings for each game. Now, you can build a neural network
# starting on line 11.
#
# Incorporate the following parameters into your model definition:
# use a sequential model
# use nine inputs and one output
# make the model dense
# use the ReLU activation function for the hidden layers
# use the linear activation function for the output layer.
#
# Train your model using both X and Y as well as the following:
# 50 epochs
# shuffle = True; this action will make Keras shuffle the data randomly during each epoch
# verbose = 2; this tells Keras to print detailed information during the processing. Take a screenshot of these
# messages for your submission.
#
# Evaluate your neural network model using model.evaluate(...) method. Print out the MSE for the test dataset.
#
# test_error_rate = model.evaluate(…)
# print( "The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
#
# Save your trained model. You will submit this model as part of your assignment.
#
# Save the model to disk  model.save("trained_model.h5")  print("Model saved to disk.")
import pandas as pd
from keras.models import Sequential
from keras.layers import *


if __name__ == '__main__':
    training_data_df = pd.read_csv("sales_data_training_scaled.csv")
    X = training_data_df.drop('total_earnings', axis=1).values
    Y = training_data_df[['total_earnings']].values

    # Define the model
    # Use a sequential model
    model = Sequential()
    # Use 9 inputs and 1 output
    # Make the model dense
    # Use the ReLU activation function for the hidden layers
    model.add(Dense(50, input_dim=9, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # Use the linear activation function for the output layer
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    # 50 epochs
    # shuffle=True; this action will make Keras shuffle the data randomly during each epoch
    # verbose = 2; this tells Keras to print detailed information during the processing.
    model.fit(X, Y, epochs=50, shuffle=True, verbose=2)

    # Load the test data
    # Load the separate test data set
    test_data_df = pd.read_csv("sales_data_testing_scaled.csv")

    X_test = test_data_df.drop('total_earnings', axis=1).values
    Y_test = test_data_df[['total_earnings']].values

    # Evaluate the model using the test data
    test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
    print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

    # Save the model to disk
    model.save("trained_model.h5")
    print("Model saved to disk.")
