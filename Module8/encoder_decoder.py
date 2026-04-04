# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Portfolio Project
# Option #2: Encoder-Decoder Model for Sequence-to-Sequence Prediction
# Part 1 (Research Write-up):
# Research and analyze the use of encoder-decoder models in industry and in applications. Your use cases should be
# uniquely distinct and should cover multiple areas of industry. Ensure that your paper meeting the following
# guidelines:
# Identify at least 4 pertinent use cases in which this model is used and the benefit for using it in each.
# Your paper should be a maximum of 4 pages and include at least 3 scholarly references in APA format. Ensure that your
# assignment is formatted according to the CSU Global Writing Center. You can easily access the Writing Center by
# clicking on the tab in the course navigation panel.
#
# Part 2 (Programming Implementation):
# For this Portfolio Project assignment, you will develop an encoder-decoder model for sequence-to-sequence prediction
# using Keras. (Keras library is a wrapper for low-level TensorFlow commands, and you will import this as a Python
# library.) The encoder-decoder model provides a pattern for using recurrent neural networks to address challenging
# sequence-to-sequence prediction problems such as machine translation.
# Encoder-decoder models can be developed in the Keras Python deep learning library. For this assignment, you will
# develop a sophisticated encoder-decoder recurrent neural network for a sequence-to-sequence prediction problem with
# Keras.
# There are three Python-based programming tasks for this Portfolio Project. The three parts are:
# 1. Encoder-Decoder Model in Keras
# 2. Scalable Sequence-to-Sequence Problem
# 3. Encoder-Decoder LSTM for Sequence Prediction.
# Implement the Python solution for this encoder-decoder specification. Include the following items in your deliverable:
# a comprehensive flowchart depicting the processing performed by your neural network model,
# the Python source file that is thoroughly documented, and
# an output text file showing the run-time predictions of your model.
#
# For your deliverable, provide a detailed analysis using your screenshots as supporting content. Write up your analysis
# using a Word document. Submit your Python code and Word document in a zip archive file. Name your archive file:
#
# CSC580_FinalPortfolio _Option_2_last_name_first_name.zip
#
# For this project, you will need the following: (Note: These are free Python packages, add-ons, and you should have no
# problem downloading these.)
# Python 3
# Scikit-learn
# Pandas
# NumPy
import numpy as np
from numpy import array, argmax, array_equal
from numpy.random import randint, seed
# Keras
# TensorFlow
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import to_categorical

seed(42)
tf.random.set_seed(42)

# Encoder-Decoder Model in Keras
# The encoder-decoder model is a way of organizing recurrent neural networks for sequence-to-sequence prediction
# problems. The approach involves two recurrent neural networks: one to encode the source sequence, called the encoder,
# and a second to decode the encoded source sequence into the target sequence, called the decoder.
# Use the following code example as a starting point to define an encoder-decoder recurrent neural network. Below is
# this function named define_models().
# returns train, inference_encoder and inference_decoder models


def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input), name='encoder_inputs')
    encoder = LSTM(n_units, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, n_output), name='decoder_inputs')
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='train_model')

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states, name='encoder_model')

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,), name='decoder_state_h')
    decoder_state_input_c = Input(shape=(n_units,), name='decoder_state_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    inf_decoder_outputs, inf_state_h, inf_state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [inf_state_h, inf_state_c]
    inf_decoder_outputs = decoder_dense(inf_decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [inf_decoder_outputs] + decoder_states, name='decoder_model'
    )
    # return all models
    return train_model, encoder_model, decoder_model

# The function takes three arguments, as follows:
# n_input: The cardinality of the input sequence, i.e., number of features, words, or characters for each time step.
# n_output: The cardinality of the output sequence, i.e., number of features, words, or characters for each time step.
# n_units: The number of cells to create in the encoder and decoder models, e.g., 128 or 256.
# The function then creates and returns three models, as follows:
#
# train: Model that can be trained if given source, target, and shifted target sequences.
# inference_encoder: Encoder model used when making a prediction for a new source sequence.
# inference_decoder: Decoder model used when making a prediction for a new source sequence.
#
# The model is trained when given source and target sequences where the model takes both the source and a shifted
# version of the target sequence as input and predicts the whole target sequence.
#
# For example, one source sequence may be [1,2,3] and the target sequence [4,5,6]. The inputs and outputs to the model
# during training would be:
#
# Input1: ['1', '2', '3']
# Input2: ['_', '4', '5']
# Output: ['4', '5', '6']
#
# The model is intended to be called recursively when generating target sequences for new source sequences.
# The source sequence is encoded and the target sequence is generated one element at a time, using a “start of sequence”
# character such as ‘_’ to start the process. Therefore, in the above case, the following input-output pairs would occur
# during training:
#
# t,   Input1,           Input2,                 Output
# 1,  ['1', '2', '3'],   '_',                          '4'
# 2,  ['1', '2', '3'],   '4',                          '5'
# 3,  ['1', '2', '3'],   '5',                          '6'
#
# Here, you can see how the recursive use of the model can be used to build up output sequences.
# During prediction, the inference_encoder model is used to encode the input sequence once, which returns states that
# are used to initialize the inference_decoder model. From that point, the inference_decoder model is used to generate
# predictions step by step.


# The function, below, named predict_sequence() can be used after the model is trained to generate a target sequence
# given a source sequence.
#
# generate target given source sequence
def predict_sequence(inf_enc, inf_dec, source, n_steps, cardinality):
    # encode
    state = inf_enc.predict(source, verbose=0)
    # start of sequence input
    # target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1)
    target_seq = np.zeros((1, 1, cardinality))
    # target_seq[0, 0, 0] = 1.0

    # collect predictions
    output = list()
    # output = []

    for t in range(n_steps):
        # predict next char
        y_hat, h, c = inf_dec.predict([target_seq] + state, verbose=0)

        # store prediction
        output.append(y_hat[0, 0, :])

        # update state
        state = [h, c]

        # update target sequence
        target_seq = y_hat
        # token = int(np.argmax(y_hat_vec))
        # target_seq = np.zeros((1, 1, cardinality), dtype=np.float32)
        # target_seq[0, 0, token] = 1.0

    return array(output)


# This function takes five arguments as follows:
#
# infenc: Encoder model used when making a prediction for a new source sequence.
# infdec: Decoder model used when making a prediction for a new source sequence.
# source: Encoded source sequence.
# n_steps: Number of time steps in the target sequence.
# cardinality: The cardinality of the output sequence, i.e., the number of features, words, or characters for each time
# step.
# The function then returns a list containing the target sequence.
#
# Scalable Sequence-to-Sequence Problem
# In this section, a contrived and scalable sequence-to-sequence prediction problem is presented for this final
# Portfolio option. The source sequence is a series of randomly generated integer values, such as
# [20, 36, 40, 10, 34, 28], and the target sequence is a reversed pre-defined subset of the input sequence, such as the
# first three elements in reverse order [40, 36, 20]. The length of the source sequence is configurable; so is the
# cardinality of the input and output sequence and the length of the target sequence. For this Portfolio Project option,
# you will use source sequences of six elements, a cardinality of 50, and target sequences of three elements.
#
# Below are some more examples to make this concrete.
#
# Source,                                                 Target
# [13, 28, 18, 7, 9, 5]                           [18, 28, 13]
# [29, 44, 38, 15, 26, 22]                  [38, 44, 29]
# [27, 40, 31, 29, 32, 1]                      [31, 40, 27]
# ...
#
# Start off by defining a function to generate a sequence of random integers. Use the value of 0 as the padding or
# start-of-sequence character; therefore, it is reserved, and you cannot use it in your source sequences. To achieve
# this, add 1 to your configured cardinality to ensure the one-hot encoding is large enough (i.e., a value of 1 maps to
# a ‘1’ value in index 1).
#
# For example:
# n_features = 50 + 1
#
# You can use the randint() Python function to generate random integers in a range between 1 and 1-minus the size of the
# problem’s cardinality.


# The generate_sequence() below generates a sequence of random integers.
# generate a sequence of random integers
def generate_sequence(length, n_unique):
    # return [randint(1, n_unique) for _ in range(length)]
    return [randint(1, n_unique-1) for _ in range(length)]


# Next, you need to create the corresponding output sequence given the source sequence. To keep things simple, select
# the first n elements of the source sequence as the target sequence and reverse them.
#
# # define target sequence
# target = source[:n_out]
# target.reverse()
#
# You also need a version of the output sequence, shifted forward by one time step, that you can use as the mock target
# generated so far, including the start-of-sequence value in the first time step. You can create this from the target
# sequence directly.
#
# # create padded input target sequence
# target_in = [0] + target[:-1]
#
# Now that all of the sequences have been defined, you can one-hot encode them, i.e., transform them into sequences of
# binary vectors. You can use the Keras built in to_categorical() function to achieve this. You can put all of this into
# a function named get_dataset() that will generate a specific number of sequences that we can use to train a model.


# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
    x_1, x_2, y_1 = list(), list(), list()
    # x_1, x_2, y_1 = [], [], []
    for _ in range(n_samples):
        # generate source sequence
        source = generate_sequence(n_in, cardinality)
        # source = generate_sequence(n_in, cardinality - 1)

        # define target sequence
        target_seq = source[:n_out]
        target_seq.reverse()
        # target = source[:n_out][::-1]

        # create padded input target sequence
        target_in = [0] + target_seq[:-1]

        # encode
        src_encoded = to_categorical([source], num_classes=cardinality)[0]
        tar_encoded = to_categorical([target_seq], num_classes=cardinality)[0]
        tar2_encoded = to_categorical([target_in], num_classes=cardinality)[0]

        # store
        x_1.append(src_encoded)
        x_2.append(tar2_encoded)
        y_1.append(tar_encoded)

    return array(x_1), array(x_2), array(y_1)


# Finally, you need to be able to decode a one-hot encoded sequence to make it readable again. This is needed for both
# printing the generated target sequences and for easily comparing whether the full predicted target sequence matches
# the expected target sequence. The one_hot_decode() function will decode an encoded sequence.
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    # return [int(argmax(vector)) for vector in encoded_seq]
    return [argmax(vector) for vector in encoded_seq]


if __name__ == '__main__':
    # Evaluate the given construction using the following code:
    # configure problem
    n_features = 50 + 1
    n_steps_in = 6
    n_steps_out = 3

    # generate a single source and target sequence
    n_train = 5000
    X1_train, X2_train, y_train = get_dataset(n_steps_in, n_steps_out, n_features, n_train)

    print(X1_train.shape, X2_train.shape, y_train.shape)
    print('X1=%s, X2=%s, y=%s' % (one_hot_decode(X1_train[0]), one_hot_decode(X2_train[0]), one_hot_decode(y_train[0])))

    # Encoder-Decoder LSTM for Sequence Prediction
    # You are now ready to apply the encoder-decoder LSTM model to a sequence-to-sequence prediction problem.

    # Define the models and compile the training model.
    # define model
    train, infenc, infdec = define_models(n_features, n_features, 128)
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Evaluate the model. Do this by making predictions for 100 source sequences and counting the number of target
    # sequences that were predicted correctly.
    # evaluate LSTM
    train.fit([X1_train, X2_train], y_train, epochs=20, batch_size=64, verbose=2)

    total, correct = 100, 0

    for _ in range(total):
        X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
        target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)

        if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
            correct += 1

    print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

    # Finally, generate some predictions and print the decoded source, target, and predicted target sequences to get an
    # idea of whether the model works as expected. An example output would be something like the following:
    # X=[22, 17, 23, 5, 29, 11] y=[23, 17, 22], yhat=[23, 17, 22]
    # X=[28, 2, 46, 12, 21, 6] y=[46, 2, 28], yhat=[46, 2, 28]
    # X=[12, 20, 45, 28, 18, 42] y=[45, 20, 12], yhat=[45, 20, 12]
    # X=[3, 43, 45, 4, 33, 27] y=[45, 43, 3], yhat=[45, 43, 3]
    # X=[34, 50, 21, 20, 11, 6] y=[21, 50, 34], yhat=[21, 50, 34]
    # X=[47, 42, 14, 2, 31, 6] y=[14, 42, 47], yhat=[14, 42, 47]
    # X=[20, 24, 34, 31, 37, 25] y=[34, 24, 20], yhat=[34, 24, 20]
    # X=[4, 35, 15, 14, 47, 33] y=[15, 35, 4], yhat=[15, 35, 4]
    # X=[20, 28, 21, 39, 5, 25] y=[21, 28, 20], yhat=[21, 28, 20]
    # X=[50, 38, 17, 25, 31, 48] y=[17, 38, 50], yhat=[17, 38, 50]
    lines = []
    for _ in range(10):
        X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
        yhat = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)

        src = one_hot_decode(X1[0])
        tgt = one_hot_decode(y[0])
        pred = one_hot_decode(yhat)

        line = f"X={src} y={tgt} yhat={pred}"
        print(line)
        lines.append(line)

    with open("predictions_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
