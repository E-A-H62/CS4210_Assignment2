# CS 4210 - Machine Learning Classifiers (Naive Bayes, k-NN, Decision Tree)

This repository contains three core programs written in Python. Each script explores a different supervised machine learning algorithm using only basic Python data structures (lists, dictionaries, arrays) and without using advanced libraries like NumPy or pandas.

---

## Project Summary

Each program aims to help students understand how standard machine learning models work in practice by:

- Reading and preprocessing data from `.csv` files manually.
- Training classifiers using `scikit-learn`.
- Performing evaluations like accuracy, confidence filtering, and error rates.

---

## File Overview

### `naive_bayes.py`
Classifies test instances using the **Naive Bayes** algorithm.

#### Description:
- Trains a Gaussian Naive Bayes model on weather data.
- Outputs test classifications **only if confidence ≥ 0.75**.

#### Input Files:
- `weather_training.csv`
- `weather_test.csv`

#### Output:
Displays predicted class and confidence score for each qualified instance.

---

### `knn.py`
Performs **Leave-One-Out Cross-Validation (LOO-CV)** on email data using a **1-Nearest Neighbor** classifier.

#### Description:
- Uses each instance as a test case while the rest are used for training.
- Calculates the overall error rate of the model.
- Classifies emails as spam or ham based on 20 feature values.

#### Input File:
- `email_classification.csv`

#### Output:
Displays the final error rate after evaluating all instances.

---

### `decision_tree_2.py`
Trains and tests three decision tree models using different training sets, measuring their average accuracy.

#### Description:
- Trains models on:
  - `contact_lens_training_1.csv`
  - `contact_lens_training_2.csv`
  - `contact_lens_training_3.csv`
- Tests all models on `contact_lens_test.csv`
- Repeats training/testing 10 times per model and averages the results.

#### Output:
Displays average accuracy per training dataset.

---

## How to Run

> 🛠️ Requirements:
- Python 3.x
- `scikit-learn` installed
- `matplotlib` only required if adding visualization (not used here)

### Install dependencies (if needed):
```bash
pip install scikit-learn
````

### Run individual scripts:

```bash
python naive_bayes.py
python knn.py
python decision_tree_2.py
```

Make sure the required `.csv` files are in the same directory as the scripts before running.

---

## Key Concepts Practiced

* Manual encoding of categorical data
* Confidence thresholding in predictions
* Leave-One-Out Cross-Validation (LOO-CV)
* Decision tree training and testing
* Evaluation of model accuracy and error rate
* Working with raw `.csv` files using Python's `csv` module
