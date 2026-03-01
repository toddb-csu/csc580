# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Module 3: Critical Thinking Assignment
# Option #1
# Option #1: Linear Regression Using TensorFlow
#
# In this assignment, you will use TensorFlow to predict the next output from a given set of random inputs.
# Start by importing the necessary libraries. You will use Numpy along with TensorFlow for computations and Matplotlib
# for plotting.
#
# For your deliverable, submit an introduction in a Word document. Submit your Python code and screenshots of your plots
# in a zip archive file. Name your archive file: CSC580_CTA_3_1_last_name_first_name.zip.
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    # In order to make the random numbers predictable, we will define fixed seeds for both Numpy and #TensorFlow.
    np.random.seed(101)
    tf.set_random_seed(101)

    # Now, let’s generate some random data for training the Linear Regression Model.
    # Generating random linear data
    # There will be 50 data points ranging from 0 to 50
    x = np.linspace(0, 50, 50)
    y = np.linspace(0, 50, 50)

    # Adding noise to the random linear data
    x += np.random.uniform(-4, 4, 50)
    y += np.random.uniform(-4, 4, 50)
    # Number of data points
    n = len(x)

    # 1) Plot the training data.
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Data')
    plt.show()

    # 2) Create a TensorFlow model by defining the placeholders X and Y so that you can feed your training examples X
    #    and Y into the optimizer during the training process.
    # X = tf.placeholder(tf.float32, name='X')
    X = tf.placeholder("float")
    # Y = tf.placeholder(tf.float32, name='Y')
    Y = tf.placeholder("float")

    # 3) Declare two trainable TensorFlow variables for the weights and bias and initialize them randomly.
    W = tf.Variable(np.random.randn(), name='W')
    b = tf.Variable(np.random.randn(), name='b')

    # 4) Define the hyperparameters for the model:
    learning_rate = 0.001
    training_epochs = 1000

    # 5) Implement Python code for: the hypothesis, the cost function, the optimizer.
    y_pred = tf.add(tf.multiply(X, W), b)
    # cost = tf.reduce_sum(tf.square(y_pred - Y)) / (2 * n)
    cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # 6) Implement the training process inside a TensorFlow session.
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            if (epoch + 1) % 100 == 0:
                c = sess.run(cost, feed_dict={X: x, Y: y})
                curr_W, curr_b = sess.run([W, b])
                print(f"Epoch: {epoch+1} cost={c:.4f} W={sess.run(W):.4f} b={sess.run(b):.4f}")

        training_cost = sess.run(cost, feed_dict={X: x, Y: y})
        weight = sess.run(W)
        bias = sess.run(b)

    # 7) Print out the results for the training cost, weight, and bias.
    print("Training Results")
    print(f"Training cost: {training_cost:.4f}")
    print(f"Weight:        {weight:.4f}")
    print(f"Bias:          {bias:.4f}")

    # 8) Plot the fitted line on top of the original data.
    plt.scatter(x, y)
    plt.plot(x, weight * x + bias, 'r', label='Fitted Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()
