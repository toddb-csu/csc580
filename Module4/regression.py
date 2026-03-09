# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Module 4: Critical Thinking Assignment
# Option #2: Logistic Regression with TensorFlow
# In this assignment, you will analyze the quality of a TensorFlow prediction by generating synthetic data.
#
# For your deliverable, provide a detailed analysis using your screenshots as supporting content. Write up your analysis
# using a Word document. Submit your Python code and Word document in a zip archive file. Name your archive file:
#
# CSC580_CTA_4_2_last_name_first_name.zip.
#
# Your paper should be a minimum of two pages in length and conform to the CSU Global Writing Center. You can easily
# access the Writing Center by clicking on the tab in the course navigation panel.
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_eager_execution()

if __name__ == '__main__':
    # In order to make the random numbers predictable, we will define fixed seeds for both Numpy and TensorFlow.
    np.random.seed(101)
    tf.set_random_seed(101)

    # 1) Generate the synthetic data using the following Python code snippet.
    # Generate synthetic data
    N = 100

    # Zeros form a Gaussian centered at (-1, -1)
    x_zeros = np.random.multivariate_normal(
        mean=np.array((-1, -1)),
        cov=.1*np.eye(2),
        size=(N//2,)
    )
    y_zeros = np.zeros((N//2,))

    # Ones form a Gaussian centered at (1, 1)
    x_ones = np.random.multivariate_normal(
        mean=np.array((1, 1)),
        cov=.1*np.eye(2),
        size=(N//2,)
    )
    y_ones = np.ones((N//2,))

    x_np = np.vstack([x_zeros, x_ones])
    y_np = np.concatenate([y_zeros, y_ones])

    # 2) Plot x_zeros and x_ones on the same graph.
    plt.scatter(x_zeros[:,0], x_zeros[:,1], c='purple', label='0', alpha=0.7)
    plt.scatter(x_ones[:,0], x_ones[:,1], c='blue', label='1', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot1.png')
    plt.show()

    tf.reset_default_graph()

    # 3)  Generate a TensorFlow graph.
    with tf.name_scope("placeholders"):
        x = tf.placeholder(tf.float32, (N, 2), name="x")
        y = tf.placeholder(tf.float32, (N,), name="y")

    with tf.name_scope("weights"):
        W = tf.Variable(tf.random.normal((2, 1)), name="W")
        b = tf.Variable(tf.random.normal((1,)), name="b")

    with tf.name_scope("prediction"):
        y_logit = tf.squeeze(tf.matmul(x, W) + b)
        # the sigmoid gives the class probability of 1
        y_one_prob = tf.sigmoid(y_logit)
        # Rounding P(y=1) will give the correct prediction.
        y_pred = tf.round(y_one_prob)

    with tf.name_scope("loss"):
        # Compute the cross-entropy term for each datapoint
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
        # Sum all contributions
        loss = tf.reduce_sum(entropy)

    with tf.name_scope("optim"):
        train_op = tf.train.AdamOptimizer(.01).minimize(loss)

    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", loss)
        merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('logistic-train', tf.get_default_graph())

    # 4) Train the model, get the weights, and make predictions.
    n_epochs = 1000
    loss_history = []

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        feed_dict = {x: x_np, y: y_np}

        # print(f"train_op type: {type(train_op)}, value: {train_op}")
        # print(f"merged type: {type(merged)}, value: {merged}")
        # print(f"loss type: {type(loss)}, value: {loss}")

        # _, loss_val = sess.run([train_op, 1], feed_dict=feed_dict)
        _, summary, loss_val = sess.run([train_op, merged, loss], feed_dict=feed_dict)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch:4d}, Loss: {loss_val:.4f}")

        train_writer.add_summary(summary, epoch)

    W_val, b_val = sess.run([W, b])
    print("Learned weights W:", W_val)
    print("Learned bias b:", b_val)

    y_pred_val = sess.run(y_pred, feed_dict={x: x_np})

    accuracy = np.mean(y_pred_val == y_np)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    train_writer.close()

    # 5) Plot the predicted outputs on top of the data.
    plt.scatter(x_np[y_np == 0, 0], x_np[y_np == 0, 1], color='blue', edgecolor='black', label='Predicted 0')
    plt.scatter(x_np[y_np == 1, 0], x_np[y_np == 1, 1], color='orange', edgecolor='black', label='Predicted 1')

    misclassified = (y_pred_val != y_np)
    if np.any(misclassified):
        plt.scatter(x_np[misclassified, 0], x_np[misclassified, 1],
                   marker='x', s=80, c='black', label='Misclassified')

    w0, w1 = W_val.flatten()
    boundary_x = np.linspace(-2.5, 2.5, 100)
    boundary_y = -(w0 / w1) * boundary_x - (b_val[0] / w1)
    plt.plot(boundary_x, boundary_y, 'k--', linewidth=2, label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('boundary_plot.png')
    plt.show()
