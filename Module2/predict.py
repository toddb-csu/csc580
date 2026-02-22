# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Module 2: Critical Thinking Assignment
# Option #2: Predicting Future Sales
# In this assignment, you will work with a neural network that can be used to predict future revenues from the sales of
# a new video game. A dataset is provided that you'll use to train a neural network to predict how much money you can
# expect future video games to earn based on historical data. The data are contained in a file named
# sales_data_training.csv,
# https://github.com/johannlilly/linkedin-learning-building-deep-learning-applications-with-keras-2-0/blob/main/exercise-files/03/sales_data_training.csv.
# In this spreadsheet, there is one row for each video game that a store has sold in the past.
#
# Next, you will load your trained model to make predictions. The prediction information is stored in
# proposed_new_product.csv and consists of one row.
#
# Complete the final segment of Python code. Be sure to rescale your final prediction using the two parameters during
# the scaling of the training and testing data sets.
#
# import pandas as pd
# from keras.models import load_model
# model = load_model('trained_model.h5')
#
# X = pd.read_csv("proposed_new_product.csv").values
# prediction = model.predict(…)
# Grab just the first element of the first prediction (since we only have one)  prediction = prediction[…][…]
# Re-scale the data from the 0-to-1 range back to dollars  # These constants are from when the data was originally
# scaled down to the 0-to-1 range  prediction = prediction + _____
# prediction = prediction / _____
# print("Earnings Prediction for Proposed Product - ${}".format(prediction))
import pandas as pd
from keras.models import load_model


if __name__ == '__main__':
    # Load the saved model from disk
    model = load_model("trained_model.h5")

    X = pd.read_csv("proposed_new_product.csv").values
    prediction = model.predict(X)

    # Grab just the first element of the first prediction (since we only have one)
    prediction = prediction[0][0]

    # Re-scale the data from the 0-to-1 range back to dollars
    # These constants are from when the data was originally scaled down to the 0-to-1 range
    prediction = prediction - 0.115913
    prediction = prediction / 0.0000036968

    print("Earnings Prediction for Proposed Product - ${}".format(prediction))
