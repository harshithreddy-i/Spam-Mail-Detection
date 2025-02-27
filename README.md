# Spam Mail Detection

## Overview
This project is a **Spam Mail Detection System** that classifies emails as **spam** or **ham (not spam)** using **Machine Learning** techniques. The model is trained on a dataset of emails with labeled spam and ham messages to automate the filtering process effectively.

## Features
- **Preprocessing**: Cleans and processes email text (removes stopwords, punctuation, and performs stemming/lemmatization).
- **Feature Extraction**: Utilizes **TF-IDF Vectorization** for better text representation.
- **Classification Model**: Uses **Naïve Bayes / Logistic Regression / SVM** for email classification.
- **Evaluation Metrics**: Measures performance with **accuracy, precision, recall, and F1-score**.
- **Deployment**: Flask web app for easy email classification.

## Tech Stack
- **Python**
- **Scikit-Learn**
- **NLTK (Natural Language Toolkit)**
- **Flask** (for Web App Deployment)
- **Pandas & NumPy** (for Data Processing)

## Dataset
The dataset used for this project consists of labeled emails with **spam** and **ham** categories. The preprocessing script cleans and transforms this dataset before feeding it into the model.

## Screenshots

### Prediction Result
![Result](https://github.com/harshithreddy-i/Spam-Mail-Detection/blob/main/accuracy%20&%20result.png?raw=true)

## Results
- **Accuracy**: ~97%
- **Precision**: 95%
- **Recall**: 96%
- **F1-Score**: 95.5%

## Future Improvements
- Improve accuracy by fine-tuning hyperparameters.
- Deploy the model as a **REST API**.
- Implement real-time spam filtering in email applications.

## Installation
### Prerequisites
Make sure you have Python installed (>=3.7). Then install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage
### 1. Train the Model
Run the following command to train the spam detection model:
```sh
python train.py
```

### 2. Test the Model
To evaluate the model on a test dataset:
```sh
python test.py
```

### 3. Run the Web Application
To launch the Flask app for email classification:
```sh
python app.py
```
Then, open `http://127.0.0.1:5000/` in your browser.


