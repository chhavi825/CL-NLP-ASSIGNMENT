# CL-NLP-ASSIGNMENT
# Computational Linguistics & Natural Language Processing

## 📌 Overview

This repository contains the LSTM Practice Questions assignment for Computational Linguistics and Natural Language Processing (CL & NLP). It covers theoretical concepts and numerical problems related to Long Short-Term Memory (LSTM) networks, sequential data processing, time-series forecasting, and classification evaluation.

The assignment focuses on understanding LSTM architecture, performing forward-pass calculations, analyzing trainable parameters, studying gradient behavior, and exploring the applications of sequential models in electricity load forecasting and predictive maintenance.

## 📚 Topics Covered

### 1. Long Short-Term Memory (LSTM)

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) designed to process sequential data and learn long-term dependencies.

The assignment covers the primary components of an LSTM cell:

- Forget Gate
- Input Gate
- Candidate Cell State
- Cell State
- Output Gate
- Hidden State

These components control how information is retained, updated, and passed through the network over time.

### 2. LSTM Forward Pass

This section involves numerical calculations for a single LSTM cell using a given input, previous hidden state, previous cell state, and gate parameters.

The calculations include:

- Forget gate activation using the sigmoid function
- Input gate activation
- Candidate cell state computation using the tanh function
- Updated cell state calculation
- Output gate activation
- Hidden state calculation

The standard LSTM equations used are:

**Forget Gate**

`f_t = sigmoid(W_f * x_t + U_f * h_(t-1) + b_f)`

**Input Gate**

`i_t = sigmoid(W_i * x_t + U_i * h_(t-1) + b_i)`

**Candidate Cell State**

`C_tilde_t = tanh(W_c * x_t + U_c * h_(t-1) + b_c)`

**Cell State**

`C_t = f_t * C_(t-1) + i_t * C_tilde_t`

**Output Gate**

`o_t = sigmoid(W_o * x_t + U_o * h_(t-1) + b_o)`

**Hidden State**

`h_t = o_t * tanh(C_t)`

These calculations demonstrate how an LSTM cell combines previous memory with new information to produce its updated internal states.

### 3. LSTM Trainable Parameter Count

This section examines the calculation of trainable parameters in an LSTM layer.

The assignment considers an LSTM with a 10-dimensional input and 32 hidden units.

The parameter calculation includes:

- Input weight matrices
- Recurrent weight matrices
- Bias vectors
- Parameters associated with all four LSTM gates
- Differences between single-bias and two-bias conventions

For input dimension `D` and hidden size `H`, the standard single-bias parameter formula is:

`P = 4H(D + H + 1)`

For a framework convention using two bias vectors per gate:

`P = 4H(D + H + 2)`

This analysis illustrates how input dimensions, hidden units, and bias configurations affect model complexity.

### 4. Vanishing Gradient Problem

The vanishing gradient problem occurs when gradients become progressively smaller during backpropagation through long sequences, making it difficult for a recurrent model to learn long-term dependencies.

The assignment investigates gradient scaling through the LSTM cell state using the forget-gate value.

The approximate gradient scaling factor is:

`Gradient Scaling Factor = f^k`

Where:

- `f` is the constant forget-gate value.
- `k` is the number of time steps.

The numerical problems examine:

- Gradient scaling for a forget-gate value of 0.9 after 10 and 50 steps
- Gradient scaling for forget-gate values of 0.5 and 0.99 after 50 steps
- The effect of forget-gate values on long-term dependency learning

The results demonstrate how the forget gate influences gradient retention and helps LSTMs learn information across longer sequences.

### 5. Time-Series Forecasting Using LSTM

This section explores the preparation of hourly electricity load data for an LSTM-based forecasting task.

The scenario considers three years of hourly electricity load records, with a 168-hour lookback window and a 24-hour forecasting horizon.

The concepts covered include:

- Total number of hourly records
- Sliding-window generation
- Lookback sequence length
- Forecasting horizon
- Input and output tensor dimensions
- Training and validation data preparation
- Mini-batch construction
- LSTM trainable parameter calculation

The input data consists of three features per time step: load, temperature, and a holiday flag.

The expected input tensor shape for a batch of 64 windows is:

`X.shape = (64, 168, 3)`

The corresponding output shape for 24-hour load forecasting is:

`y.shape = (64, 24)`

These calculations demonstrate how sequential observations are transformed into structured input and output tensors for time-series forecasting.

### 6. Min-Max Normalization

Min-Max normalization rescales numerical values to a specified range, commonly between 0 and 1.

**Formula:**

`X_scaled = (X - X_min) / (X_max - X_min)`

Where:

- `X` — Original data value
- `X_min` — Minimum value
- `X_max` — Maximum value
- `X_scaled` — Normalized value

The assignment applies this transformation to vibration sensor readings ranging from 20 mm/s to 120 mm/s.

Normalization helps represent numerical values on a consistent scale for machine learning workflows.

### 7. Predictive Maintenance and Failure Prediction

This section explores the use of an LSTM model to predict whether industrial equipment will fail within the next operating cycle.

The scenario involves vibration sensor readings collected during calibration and the prediction of equipment failure using a probability threshold.

The assignment covers:

- Min-Max normalization of sensor readings
- Failure probability interpretation
- Binary classification using a decision threshold of 0.5
- Identification of predicted failures
- Evaluation of predictions against actual outcomes

A predicted probability greater than or equal to the threshold is classified as a positive failure prediction.

This scenario illustrates how sequential models can support predictive maintenance by identifying equipment that may require inspection.

## 📊 Model Evaluation

The assignment evaluates binary classification performance using a confusion matrix.

The confusion matrix consists of four components:

- **True Positive (TP):** Actual failures correctly predicted as failures.
- **True Negative (TN):** Actual non-failures correctly predicted as non-failures.
- **False Positive (FP):** Non-failures incorrectly predicted as failures.
- **False Negative (FN):** Actual failures incorrectly predicted as non-failures.

The following evaluation metrics are calculated:

### Precision

`Precision = TP / (TP + FP)`

Measures the proportion of predicted failures that are actual failures.

### Recall

`Recall = TP / (TP + FN)`

Measures the proportion of actual failures correctly identified by the model.

### F1-Score

`F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

Combines precision and recall using their harmonic mean.

### Accuracy

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

Measures the proportion of all correct predictions.

The assignment uses a test set of 1,000 operating cycles, with 40 actual failures, 45 predicted failures, and 30 correctly identified failures.

The evaluation highlights the importance of interpreting precision, recall, F1-score, and accuracy together.

In predictive maintenance, recall is particularly important because missed failures can lead to equipment damage, operational downtime, and safety risks. Accuracy alone may be misleading when the dataset contains many more normal operating cycles than failure cases.

## 🧠 Key Learning Outcomes

After completing this assignment, the following concepts can be understood:

- Fundamentals and architecture of LSTM networks
- Functionality of LSTM gates and memory states
- Numerical computation of an LSTM forward pass
- Calculation of trainable parameters
- Differences between LSTM bias conventions
- Gradient propagation and the vanishing gradient problem
- Influence of forget-gate values on gradient retention
- Sequential data preparation for time-series forecasting
- Sliding-window generation and tensor dimensions
- Training windows and mini-batch construction
- Min-Max normalization of numerical data
- Probability thresholds for binary classification
- Confusion matrix interpretation
- Precision, recall, F1-score, and accuracy
- Evaluation of classification models in predictive maintenance

## 🔍 Applications of LSTM

LSTM networks are widely used in sequential learning tasks, including:

- Natural Language Processing
- Text Generation and Prediction
- Sentiment Analysis
- Machine Translation
- Speech Recognition
- Time-Series Forecasting
- Electricity Load Prediction
- Predictive Maintenance

## 👩‍💻 Author

**Chhavi Chauhan**

B.Tech — Computer Science (AI & ML)

Computational Linguistics & Natural Language Processing Assignment

## 📄 Disclaimer

This repository is intended for academic and educational purposes as part of coursework in Computational Linguistics and Natural Language Processing. The content focuses on LSTM architecture, numerical problem-solving, sequential data processing, time-series forecasting, and classification evaluation.
