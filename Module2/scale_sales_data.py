# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Module 2: Critical Thinking Assignment
# Option #2: Predicting Future Sales
#
# In this assignment, you will work with a neural network that can be used to predict future revenues from the sales of
# a new video game. A dataset is provided that you'll use to train a neural network to predict how much money you can
# expect future video games to earn based on historical data. The data are contained in a file named
# sales_data_training.csv,
# https://github.com/johannlilly/linkedin-learning-building-deep-learning-applications-with-keras-2-0/blob/main/exercise-files/03/sales_data_training.csv.
# In this spreadsheet, there is one row for each video game that a store has sold in the past.
#
# (Source: Geitgey, A.(n.d.). Building deep learnging: Keras Tree Master 03,
# https://github.com/johannlilly/linkedin-learning-building-deep-learning-applications-with-keras-2-0/tree/main/exercise-files/03.
# The columns are defined as follows:
# critic_rating: an average star rating out of five stars.
# is_action: tells us if this was an action game.
# is_exclusive_to_us: tells us if we have an exclusive deal to sell this game.
# is_portable: tells us if this game runs on a handheld video game system.
# is_role_playing: tells us if this is a role - playing game, which is a genre of video game.
# is_sequel: tells us if this game was a sequel to an earlier video game and part of an ongoing series.
# is_sports: tell us if this was a sports game in the sports genre.
# suitable_for_kids tells us if this game is appropriate for all ages.
# total_earnings: tells us how much money the store has earned in total from selling the game to all customers.
# unit_price tells us for how much a single copy of the game retailed.
#
# You’ll use Keras to train the neural network that will try to predict the total earnings of a new game based on these
# characteristics.Along with the sales_data_training.csv file, there is also a second data file called
# sales_data_test.csv,
# https://github.com/johannlilly/linkedin-learning-building-deep-learning-applications-with-keras-2-0/blob/main/exercise-files/03/sales_data_test.csv.
# This file is exactly like the first one. The machine learning system should only use the training dataset during the
# training phase. Then, you'll use the test data to check how well the neural network is working. To use this data to
# train a neural network, you first have to scale this data so that each value is between zero and one. Neural networks
# train best when data in each column is all scaled to the same range. Use the following Python code to scale the
# earning and unit price columns in both the training and test datasets. You will use Pandas to generate scaled
# training and test data sets.
#
# import pandas as pdfrom
# sklearn.preprocessing
# import MinMaxScaler
# Load training data set from CSV filetraining_data_df = pd.read_csv("sales_data_training.csv")
# Load testing data set from CSV filetest_data_df = pd.read_csv("sales_data_test.csv")
# Data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
# scaler = MinMaxScaler(feature_range=(0, 1))
# Scale both the training inputs and outputs
# scaled_training = scaler.fit_transform(training_data_df)
# scaled_testing = scaler.transform(test_data_df)
# Print out the adjustment that the scaler applied to the total_earnings column of dataprint
# ("Note: total_earnings values were scaled by multiplying by {:.10f} and
# adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))
# Create new pandas DataFrame objects from the scaled data
# scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
# scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)
# Save scaled data dataframes to new CSV files
# scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
# scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    # Load training data set from CSV file
    training_data_df = pd.read_csv("sales_data_training.csv")
    # Load testing data set from CSV file
    test_data_df = pd.read_csv("sales_data_test.csv")
    # Data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale both the training inputs and outputs
    scaled_training = scaler.fit_transform(training_data_df)
    scaled_testing = scaler.transform(test_data_df)
    # Print out the adjustment that the scaler applied to the total_earnings column of data
    print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))
    # Create new pandas DataFrame objects from the scaled data
    scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
    scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)
    # Save scaled data dataframes to new CSV files
    scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
    scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)
