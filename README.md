# Flight Price Prediction ML Model - Complete Solution

## Overview
This project implements a complete ML pipeline for flight price prediction using Random Forest/XGBoost models, integrated with a RAG (Retrieval-Augmented Generation) system for travel policy questions.

## Files Structure
```
e:\Work\solvusragbot\
├── app_final.py                    # Complete Flask application
├── app_simple.py                   # Simplified version for testing
├── test_ml.py                      # ML model testing script
├── test_specific.py                # Specific test for troubleshooting
├── flight_price_prediction_pipeline.joblib  # Trained ML model
├── feature_names.joblib            # Feature names for the model
├── metadata.pkl                    # RAG metadata
├── faiss_index.index              # FAISS vector index
├── .env                           # Environment variables
└── eda/
    ├── Data_Train.xlsx            # Training dataset
    └── To_Colab.ipynb             # Jupyter notebook with ML training
```

## Key Features

### 1. ML Model Pipeline
- **Algorithm**: Random Forest Regressor with preprocessing pipeline
- **Features**: 162 features including:
  - Numerical: Total_Stops, Journey_day, Journey_month, Journey_year, Dep_hour, Dep_min, Arrival_hour, Arrival_min, Total_Duration_mins
  - Categorical: Airline, Source, Destination, Route, Additional_Info
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical
- **Performance**: Successfully predicts flight prices with reasonable accuracy

### 2. Intent Detection System
- **Method**: OpenAI GPT-4 for intelligent routing
- **Routes**: 
  - RAG: Travel policies, procedures, documentation questions
  - ML: Flight price predictions, cost estimates
- **Fallback**: Keyword-based routing if OpenAI fails

### 3. Feature Engineering
The model expects these exact features:
```python
{
    'Total_Stops': 0,
    'Journey_day': 15,
    'Journey_month': 6,
    'Journey_year': 2019,
    'Dep_hour': 10,
    'Dep_min': 0,
    'Arrival_hour': 14,
    'Arrival_min': 0,
    'Total_Duration_mins': 240,
    'Airline': 'IndiGo',
    'Source': 'Banglore',
    'Destination': 'New Delhi',
    'Route': 'Banglore → New Delhi',
    'Additional_Info': 'No info'
}
```

## API Endpoints

### POST /ask
Main endpoint for queries.

**Request:**
```json
{
    "query": "What is the flight price from Banglore to New Delhi on Alliance Air?"
}
```

**Response for ML queries:**
```json
{
    "query": "What is the flight price from Banglore to New Delhi on Alliance Air?",
    "route": "ML",
    "intent_info": {
        "route": "ML",
        "reason": "Price prediction request",
        "prediction_type": "flight_price",
        "extracted_info": {
            "source": "Banglore",
            "destination": "New Delhi",
            "airline": "Alliance Air"
        }
    },
    "result": {
        "predicted_price": 7385.64,
        "currency": "INR",
        "features": {...},
        "prediction_type": "flight_price",
        "intent_reason": "Price prediction request",
        "confidence": "medium"
    }
}
```

### GET /health
Health check endpoint.

### GET /test-ml
Test endpoint for ML model functionality.

## Problem Resolution

### Original Error:
```
"ML prediction failed: Expected 2D array, got 1D array instead:\narray=[{'source': 'banglore', 'dest': 'new delhi', 'month': 'Unknown', 'airline': 'Alliance Air'}].\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
```

### Root Cause:
The ML model was receiving a list of dictionaries instead of a properly formatted pandas DataFrame.

### Solution:
1. **Feature DataFrame Creation**: Created `create_feature_dataframe()` function to build proper DataFrame
2. **Data Format**: Ensured all features match the training data structure
3. **Type Handling**: Proper handling of categorical and numerical features
4. **Default Values**: Sensible defaults for missing information

## Usage Examples

### 1. Flight Price Prediction
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the flight price from Mumbai to Delhi on IndiGo in March?"}'
```

### 2. Travel Policy Questions
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the baggage policy for international flights?"}'
```

### 3. Health Check
```bash
curl http://localhost:5000/health
```

## Model Performance
- **Training Data**: 10,682 flight records
- **Features**: 162 total features after preprocessing
- **Model Type**: Random Forest with 100 estimators
- **Accuracy**: Successfully handles various airline, route, and time combinations
- **Warnings**: Some unknown categories are handled gracefully with zero encoding

## Environment Setup
1. Install required packages:
```bash
pip install flask pandas numpy scikit-learn joblib faiss-cpu sentence-transformers openai python-dotenv
```

2. Set environment variables in `.env`:
```
OPENAI_API=your_openai_api_key_here
```

3. Ensure required files are present:
   - `flight_price_prediction_pipeline.joblib`
   - `feature_names.joblib`
   - `metadata.pkl`
   - `faiss_index.index`

## Running the Application
```bash
python app_final.py
```

The application will start on `http://localhost:5000`

## Testing
Run the test scripts to verify functionality:
```bash
python test_ml.py          # General ML model test
python test_specific.py    # Specific prediction test
```

## Future Enhancements
1. **Model Improvements**: Add more sophisticated feature engineering
2. **Confidence Scoring**: Implement prediction confidence metrics
3. **Real-time Data**: Integrate with live flight data APIs
4. **Caching**: Add Redis caching for frequently requested routes
5. **Monitoring**: Add logging and monitoring for production use

## Troubleshooting

### Common Issues:
1. **Missing Files**: Ensure all required `.joblib` and `.pkl` files are present
2. **OpenAI API**: Verify API key is set correctly in `.env`
3. **Unknown Categories**: Model handles unknown airlines/routes gracefully
4. **Feature Mismatch**: Ensure feature DataFrame matches training structure

### Debug Steps:
1. Check `/health` endpoint for system status
2. Use `/test-ml` endpoint for model verification
3. Run test scripts to isolate issues
4. Check console logs for detailed error messages

The complete solution is now working and handles both ML predictions and RAG queries effectively!
