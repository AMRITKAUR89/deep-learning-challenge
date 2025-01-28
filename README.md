# deep-learning-challenge

# Neural Network Model Report

## [Overview](#overview)
The purpose of this analysis was to build and evaluate a deep learning model for a binary classification task using a charity dataset. The goal was to predict a target variable based on various input features related to different types of charity organizations. We aimed to preprocess the data, build a neural network, train the model, and evaluate its performance to achieve a target accuracy of more than 75%.

## [Data Preprocessing](#data-preprocessing)

- **Target Variable(s):**  
  The target variable for this model is the `IS_SUCCESSFUL` column, which indicates whether a charity application is successful or not (binary classification).
  
- **Feature Variable(s):**  
  The features used for the model are all the remaining columns in the dataset, after dropping irrelevant columns (such as `EIN` and `NAME`). These features include categorical variables like `APPLICATION_TYPE`, `CLASSIFICATION`, and `USE_CASE`, as well as numerical variables such as `AFFILIATION`, `ORGANIZATION`, and more.

- **Variables to Remove:**  
  The `EIN` and `NAME` columns were removed because they contain identifiers that do not contribute to the predictive modeling process and are not relevant to either the target or the features.

## [Model Compilation](#model-compiling)

- **Neurons, Layers, and Activation Functions:**  
  - **Input Layer:**  
    The first hidden layer has 128 neurons, with `ReLU` activation. The number of neurons corresponds to the number of features in the dataset (`X_train.shape[1]`).
  
  - **Hidden Layers:**  
    The model has three hidden layers:
    - **First hidden layer:** 128 neurons with `ReLU` activation.
    - **Second hidden layer:** 64 neurons with `ReLU` activation.
    - **Third hidden layer:** 32 neurons with `ReLU` activation.
  
  - **Dropout Layers:**  
    Dropout layers were included after each hidden layer with a 20% rate to prevent overfitting.
  
  - **Batch Normalization:**  
    Added after each hidden layer to stabilize training.
  
  - **Output Layer:**  
    A single neuron with `sigmoid` activation for binary classification (0 or 1).

- **Model Performance:**  
  After training for 50 epochs, the model achieved a training accuracy above 75%. However, the performance fluctuated due to potential overfitting or insufficient epochs.

- **Steps Taken to Improve Performance:**  
  - **Model Architecture:** We started with a relatively simple architecture (three hidden layers), but more layers and neurons could be added if performance is insufficient.
  - **Dropout:** Dropout layers were included to avoid overfitting by randomly "dropping" a fraction of the neurons during each training iteration.
  - **Batch Normalization:** Added to help improve training stability.
  - **Optimizer:** The Adam optimizer was used, as it is well-suited for most tasks and adjusts the learning rate during training.
  - **Early Stopping:** Consider using early stopping to halt training when validation accuracy does not improve after a set number of epochs.

## [Summary](#summary)

- **Model Results:**  
  The neural network achieved an accuracy of over 75% on the training set and reasonable performance on the validation set. The model showed promising results, but there is still room for improvement with further hyperparameter tuning, cross-validation, and feature engineering.

- **Recommendation for Improvement:**  
  A **Random Forest Classifier** could be a good alternative for this classification problem. It is a robust, ensemble-based algorithm that can handle both categorical and continuous variables well and is less prone to overfitting than neural networks in some cases. Additionally, Random Forest models offer feature importance, which could help identify the most important variables for prediction.

- **Explanation:**  
  Random Forests generally require less tuning and can perform well on datasets with many features. They do not need to be trained as extensively as neural networks, and the interpretability (through feature importance) is beneficial for understanding which features influence the prediction.

## Visual Support

Below is a plot showing the training and validation accuracy over the epochs:

```python
import matplotlib.pyplot as plt
# Plot accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

