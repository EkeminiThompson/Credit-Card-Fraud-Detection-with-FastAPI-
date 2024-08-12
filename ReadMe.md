# Real-Time Credit Card Fraud Detection with FastAPI and Machine Learning

This project implements a real-time credit card fraud detection system using machine learning, deployed with FastAPI. The system predicts the likelihood of fraudulent transactions based on various features extracted from the transaction data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Credit card fraud is a significant issue in the financial industry. This project aims to create a system that can detect fraudulent transactions in real-time using a trained machine learning model. The model is deployed using FastAPI, allowing for efficient and scalable predictions.

## Features

- **Real-Time Fraud Detection**: The system provides instant feedback on whether a transaction is likely to be fraudulent.
- **Scalable API**: FastAPI ensures that the model can handle multiple requests concurrently.
- **Machine Learning**: A Random Forest classifier is used for predicting fraud based on transaction features.
- **Data Preprocessing**: The model uses standardized features to improve accuracy.

## Technologies Used

- **Python**: Core programming language.
- **FastAPI**: Framework for building the web API.
- **Scikit-Learn**: Library used for machine learning model training.
- **Joblib**: For model serialization and deserialization.
- **Pandas & NumPy**: Data manipulation and preprocessing.
- **Google Colab**: For model training and experimentation.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager
- Virtual environment (optional but recommended)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI application:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the API:**
   Open your browser and go to `http://127.0.0.1:8000/docs` to view the interactive API documentation.

## Usage

### Making Predictions

You can use `curl`, Postman, or any other API testing tool to make predictions. Here's an example `curl` command:

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
-H "Content-Type: application/json" \
-d '{
    "V1": -1.359807134,
    "V2": -0.072781173,
    ...
    "Amount": 149.62
}'
```

### Response
```json
{
    "fraud_prediction": 0,
    "probability": 0.01
}
```

## Model Training

The model is trained on the Credit Card Fraud Detection Dataset from Kaggle. The dataset is highly imbalanced, so techniques like oversampling were used to balance the classes. The trained model and scaler are serialized using `joblib` for deployment.

### Steps:

1. **Preprocess the data:**
   - Scale features using StandardScaler.
   - Balance the dataset using oversampling.

2. **Train the model:**
   - A Random Forest classifier is used.

3. **Save the model and scaler:**
   - Serialize them with `joblib` for later use in the API.

## API Endpoints

- `POST /predict/`: Predict whether a transaction is fraudulent. Expects a JSON payload with the transaction features.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
