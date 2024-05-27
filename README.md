# Tree Species Classification using Neural Networks

## Project Overview

This project aims to classify three species of trees (Piper longum, Piper betel, and Piper nigrum) using a neural network model based on various leaf and stem characteristics. The model is built using TensorFlow and Keras and includes steps for data preprocessing, model training, evaluation, and prediction.


## Group Details
- MAF Amana
- MIF Azra
- MS Zainab

  
## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
  

## Dataset

The dataset includes various features of tree leaves and stems, such as:

- Leaf length (cm)
- Leaf width (cm)
- Stem diameter (cm)
- Internode length (cm)
- Leaf vein density (veins/cm²)
- Number of leaves per node

The data is divided into training and testing sets.

## Data Preprocessing

### Balancing the Dataset

The training dataset is balanced by resampling the minority class to ensure an equal number of samples across all classes.

### Standardizing Features

Features are standardized to have a mean of 0 and a standard deviation of 1 using Scikit-learn's `StandardScaler`.

## Model Architecture

The neural network model consists of:

- Input layer: Dense layer with 64 neurons and ReLU activation
- Hidden layer: Dense layer with 32 neurons and ReLU activation
- Output layer: Dense layer with 3 neurons and softmax activation

## Training the Model

The model is compiled using:

- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`

The model is trained for 50 epochs with a batch size of 16, and 20% of the training data is used for validation.

## Evaluation

The model is evaluated on the testing dataset to compute test loss and accuracy. A classification report and confusion matrix are generated to assess the model's performance.

## Prediction

The model can predict the species of a tree based on user input for the various leaf and stem characteristics.

## Installation and Usage

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- Joblib

### Clone the repository

```bash
git clone https://github.com/amanaazmeer/ICT3212_Tree-Species-Classification.git```

### Running the Code

1. Load the data:

    ```python
    training_data = pd.read_csv('TrainingDataset.csv')
    testing_data = pd.read_csv('TestingDataSet.csv')
    ```

2. Preprocess the data:

    ```python
    X_train, y_train, scaler = preprocess_data(training_data)
    X_test, y_test, _ = preprocess_data(testing_data, scaler, is_train=False)
    ```

3. Train the model:

    ```python
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
    ```

4. Evaluate the model:

    ```python
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')
    ```

5. Save the model and scaler:

    ```python
    model.save('trained_model.h5')
    dump(scaler, 'scaler.joblib')
    ```

6. Make predictions:

    ```python
    input_data = { ... }
    input_scaled = preprocess_user_input(input_data, scaler)
    prediction = model.predict(input_scaled)
