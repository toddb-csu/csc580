# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Portfolio Project Milestone
# Option #2: Distinguishing Dogs and Cats
# In this assignment, you will train a Convolutional Neural Network (CNN) to predict whether an image contains a dog or
# a cat. To do this, use Kaggle’s cats and dogs Dataset: https://www.kaggle.com/c/dogs-vs-cats
# (Click on “Data” at the top of the homepage.)
# It contains 12,500 pictures of cats and 12,500 of dogs, with different resolutions.
#
# Submit your Python code and Word document in a zip archive file. Name your archive file:
# CSC580_ CTA6_Option_2_last_name_first_name.zip
# Format your paper according to the guidelines in the CSU Global Writing Center.
# You can easily access the Writing Center by clicking on the tab in the course navigation panel.

# Step 1: Load and preprocess the image data with NumPy.
import tensorflow as tf
import seaborn as sns
import numpy as np
import glob
import matplotlib.pyplot as plt

from PIL import Image
from collections import defaultdict
# The following code implements a CNN using a single hidden layer.
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

# The most common shape size is 375×500. Divide this by four for your network’s input.
# Use this for the image loading code.
IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT = 94, 125
IMG_SIZE = (IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT)


def pixels_from_path(file_path):
    im = Image.open(file_path)
    im = im.convert("RGB")
    im = im.resize((IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT))
    np_im = np.asarray(im, dtype=np.float32) / 255.0

    # matrix of pixel RGB values
    return np_im


def animal_pic(index):
    # return Image.fromarray(x_valid[index])
    return Image.fromarray((x_valid[index] * 255).astype(np.uint8))


def cat_index(index, model):
    # return conv_model.predict(np.asarray([x_valid[124]]))[0][0]
    return model.predict(np.asarray([x_valid[index]]), verbose=0)[0][0]
    # return float(model.predict(x_valid[index][None, ...], verbose=0)[0][0])


def get_prediction(i, model):
    image = animal_pic(i)
    # Display Image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Get Prediction
    # probability = model.predict(image)[0][0]
    # probability = model.predict(x_valid[i][None, ...], verbose=0)[0][0]
    probability = model.predict(np.expand_dims(x_valid[i], axis=0), verbose=0)[0][0]

    atype = "CAT" if probability > 0.5 else "DOG"
    print(f"Index: {i}")
    print(f"Probability of being a CAT: {probability:.4f}")
    print(f"Model Prediction: This is a {atype}")
    print("probability of being a cat: {}".format(cat_index(i, model)))
    # probability of being a cat: 0.046173080801963806


if __name__ == '__main__':
    shape_counts = defaultdict(int)

    for i, cat in enumerate(glob.glob('cats/*')[:1000]):
        if i % 100 == 0:
            print(i)

        img_shape = pixels_from_path(cat).shape
        shape_counts[str(img_shape)]= shape_counts[str(img_shape)]+ 1
    shape_items = list(shape_counts.items())
    shape_items.sort(key=lambda x: x[1])
    shape_items.reverse()
    print(shape_items[:5])

    # 10% of the data will automatically be used for validation
    validation_size = 0.1
    # resize images to be 374x500 (most common shape)
    img_size = IMG_SIZE
    # RGB
    num_channels = 3
    # We'll use 8192 pictures (2**13)
    sample_size = 8192
    print(pixels_from_path(glob.glob('cats/*')[5]).shape)

    # Resolve which sample size is the right sample size
    SAMPLE_SIZE = sample_size # 2048
    print("loading training cat images...")
    cat_train_set = np.asarray([pixels_from_path(cat) for cat in glob.glob('cats/*')[:SAMPLE_SIZE]])
    print("loading training dog images...")
    dog_train_set = np.asarray([pixels_from_path(dog) for dog in glob.glob('dogs/*')[:SAMPLE_SIZE]])

    valid_size = 512
    print("loading validation cat images...")
    cat_valid_set = np.asarray([pixels_from_path(cat) for cat in glob.glob('cats/*')[-valid_size:]])
    print("loading validation dog images...")
    dog_valid_set = np.asarray([pixels_from_path(dog) for dog in glob.glob('dogs/*')[-valid_size:]])

    x_train = np.concatenate([cat_train_set, dog_train_set])
    labels_train = np.asarray([1 for _ in range(SAMPLE_SIZE)]+[0 for _ in range(SAMPLE_SIZE)])
    x_valid = np.concatenate([cat_valid_set, dog_valid_set])
    labels_valid = np.asarray([1 for _ in range(valid_size)]+[0 for _ in range(valid_size)])

    total_pixels = img_size[0] * img_size[1] * 3
    fc_size = 512
    inputs = keras.Input(shape=(img_size[1], img_size[0], 3), name='ani_image')
    # inputs = keras.Input(shape=(img_size[1], img_size[0], 3), name='ani_image')
    # turn image to vector.
    x = layers.Flatten(name='flattened_img')(inputs)
    x = layers.Dense(fc_size, activation='relu', name='first_layer')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='class')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Step 2: Do the following for this model:
    # ·         Use the AdamOptimizer,
    # ·         Use 10 epochs,
    # ·         Shuffle the training data, and
    # ·         Use MSE as the loss function.
    customAdam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=customAdam,  # Optimizer
                  # Loss function to minimize
                  loss="mean_squared_error",
                  # List of metrics to monitor
                  metrics=["binary_crossentropy", "mean_squared_error"])

    print('# Fit model on training data')

    history = model.fit(x_train,
                        labels_train,
                        batch_size=32,
                        # important since we loaded cats first, dogs second.
                        shuffle=True,
                        epochs=10,
                        validation_data=(x_valid, labels_valid))

    # Train on 4096 samples, validate on 2048 samples
    # loss: 0.5000 - binary_crossentropy: 8.0590 - mean_squared_error: 0.5000 - val_loss: 0.5000 - val_binary_crossentropy: 8.0591 - val_mean_squared_error: 0.5000

    # Step 3: Train the CNN. Use the following configuration for the network:
    # ·         One single convolution layer with 24 kernels,
    # ·         Two fully connected layers,
    # ·         Max pooling,
    # ·         Measure the Pearson correlation between predictions and validation labels.
    fc_layer_size = 128
    img_size = IMG_SIZE

    conv_inputs = keras.Input(shape=(img_size[1], img_size[0],3), name='ani_image')
    conv_layer = layers.Conv2D(24, kernel_size=3, activation='relu')(conv_inputs)
    conv_layer = layers.MaxPool2D(pool_size=(2,2))(conv_layer)
    conv_x = layers.Flatten(name = 'flattened_features')(conv_layer) #turn image to vector.

    conv_x = layers.Dense(fc_layer_size, activation='relu', name='first_layer')(conv_x)
    conv_x = layers.Dense(fc_layer_size, activation='relu', name='second_layer')(conv_x)
    conv_outputs = layers.Dense(1, activation='sigmoid', name='class')(conv_x)

    conv_model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)

    customAdam = keras.optimizers.Adam(learning_rate=1e-4)
    conv_model.compile(optimizer=customAdam,  # Optimizer
                  # Loss function to minimize
                  loss="binary_crossentropy",
                  # List of metrics to monitor
                  metrics=["binary_crossentropy","mean_squared_error"])

    #Epoch 5/5 loss: 1.6900 val_loss: 2.0413 val_mean_squared_error: 0.3688
    print('# Fit model on training data')

    history = conv_model.fit(x_train,
                        labels_train, #we pass it th labels
                        #If the model is taking too long to train, make this bigger
                        #If it is taking too long to load for the first epoch, make this smaller
                        batch_size=32,
                        shuffle = True,
                        epochs=10,
                        # We pass it validation data to
                        # monitor loss and metrics
                        # at the end of each epoch
                        validation_data=(x_valid, labels_valid))

    preds = conv_model.predict(x_valid)
    preds = np.asarray([pred[0] for pred in preds])
    print("Pearson r: ", np.corrcoef(preds, labels_valid)[0][1]) # 0.15292172

    # Step 4: Perform the following analysis:
    # 1) Modify the model by adding another convolutional layer and use 48 kernels.
    # What is the new correlation coefficient?
    conv2_inputs = keras.Input(shape=(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, 3), name='ani_image')

    x = layers.Conv2D(24, 3, activation='relu', padding='same', name='conv1')(conv2_inputs)
    x = layers.MaxPool2D((2, 2), name='pool1')(x)

    x = layers.Conv2D(48, 3, activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPool2D((2, 2), name='pool2')(x)

    x = layers.Flatten(name='flattened_features')(x)
    x = layers.Dense(fc_layer_size, activation='relu', name='fc1')(x)
    x = layers.Dense(fc_layer_size, activation='relu', name='fc2')(x)
    out = layers.Dense(1, activation='sigmoid', name='class')(x)

    conv2_model = keras.Model(conv2_inputs, out)

    conv2_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", "mean_squared_error"]
    )

    history2 = conv2_model.fit(
        x_train, labels_train,
        batch_size=32,
        shuffle=True,
        epochs=10,
        validation_data=(x_valid, labels_valid)
    )

    preds2 = conv2_model.predict(x_valid, batch_size=32, verbose=0).ravel()
    pearson_r2 = np.corrcoef(preds2, labels_valid)[0, 1]
    print("New correlation coefficient Pearson r2:", pearson_r2)

    # 2) Assess the accuracy of the model.
    sns.scatterplot(x=preds, y=labels_valid)
    plt.title("Conv Model")
    plt.show()
    sns.scatterplot(x=preds2, y=labels_valid)
    plt.title("Conv2 Model")
    plt.show()

    cat_quantity = sum(labels_valid)

    for i in range(1, 10):
        threshold = 0.1 * i
        # predictions above threshold
        mask = (preds2 > threshold)
        precision = np.sum(labels_valid[mask]) / np.sum(mask) if np.sum(mask) > 0 else 0
        # I'm not sure what code was missing in what was copied from the assignment
        # print(sum(labels_valid[preds  .1*i])/labels_valid[preds  .1*i].shape[0])
        print(f'threshold :{threshold:.1f} - Accuracy/Precision: {precision:.4f}')

    # Overall accuracy at 0.5 threshold
    threshold = 0.5
    predicted_labels = (preds2 > threshold).astype(int)
    accuracy = np.mean(predicted_labels == labels_valid)
    print(f"Overall Accuracy at 0.5 threshold: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels_valid, predicted_labels, target_names=['Dog', 'Cat']))

    # Accuracy at different thresholds
    for i in range(1, 10):
        threshold = 0.1 * i
        binary_preds = (preds2 >= threshold).astype(int)
        accuracy = np.mean(binary_preds == labels_valid)
        print(f'Threshold {threshold:.1f}: Accuracy = {accuracy:.4f}')

    # Step 5: Interpret the scatterplot of the summary result.
    # Use the following utility functions to select an image and report the probability of the image being a cat.
    # An example output would be:
    index = 600
    get_prediction(index, model=conv_model)

    # Step 6: Save the model.
    conv_model.save('my_conv_model')

    # Step 7: Write an interface to your model that allows the user to pick an index and prints out the image and the
    # probability of the image being one of a cat. In a Word document, provide an analysis of the veracity of your
    # model. Suggest ways of improving your model.
    while True:
        user_input = input("Pick an number to display the image and probability that it is a cat or enter 'q' to quit:")
        if user_input.lower() == 'q':
            print("Goodbye!")
            break

        try:
            idx = int(user_input)
            if idx < 0 or idx >= len(x_valid):
                print("Value must be between 0 and {}".format(len(x_valid)))
                continue
            get_prediction(idx, model=conv_model)
        except ValueError:
            continue
