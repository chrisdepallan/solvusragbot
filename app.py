from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import faiss
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

# ==== Configuration ====
load_dotenv()  # Load environment variables from .env file

client = OpenAI(api_key=os.getenv("OPENAI_API"))
app = Flask(__name__)

# ==== Load RAG ====
try:
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    with open("all_chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
        
    faiss_index = faiss.read_index("faiss_index.index")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("RAG system loaded successfully")
    rag_loaded = True
except Exception as e:
    print(f"RAG system failed to load: {e}")
    rag_loaded = False
    metadata = []
    all_chunks = []
    faiss_index = None
    embed_model = None

# ==== Load ML Pipeline ====
try:
    ml_pipeline = joblib.load("flight_price_prediction_pipeline.joblib")
    feature_names = joblib.load("feature_names.joblib")
    print("ML model loaded successfully")
    ml_loaded = True
except FileNotFoundError as e:
    print(f"ML model files not found: {e}")
    ml_pipeline = None
    feature_names = None
    ml_loaded = False

# ==== SIMPLE ROUTING ====
def route_query(query):
    """
    Simple keyword-based routing
    """
    query_lower = query.lower()
    
    # ML keywords
    ml_keywords = [
        "predict", "estimate", "cost", "price", "fare", "how much", 
        "flight from", "travel cost", "ticket price", "airfare"
    ]
    
    if any(keyword in query_lower for keyword in ml_keywords):
        return "ML"
    else:
        return "RAG"

# ==== RAG HANDLER ====
def handle_rag_query(query):
    """
    Handle RAG queries
    """
    if not rag_loaded:
        return {
            "answer": "RAG system is not available. Please check that metadata.pkl, all_chunks.pkl, and faiss_index.index files are present.",
            "error": True
        }
    
    try:
        # Retrieve relevant chunks
        q_embed = embed_model.encode([query])
        D, I = faiss_index.search(q_embed, 3)
        
        # Get context from retrieved chunks
        context = "\n\n".join([all_chunks[i] for i in I[0] if i < len(all_chunks)])
        
        # Generate response using OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful travel assistant. Answer questions based on the provided travel policy documents."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": len(I[0])
        }
        
    except Exception as e:
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "error": True
        }

# ==== ML HANDLER ====
def extract_features_from_query(query):
    """
    Extract flight features from natural language query
    """
    # Default values
    features = {
        'Total_Stops': 0,
        'Journey_day': 15,
        'Journey_month': 6,
        'Journey_year': 2024,
        'Dep_hour': 10,
        'Dep_min': 0,
        'Arrival_hour': 14,
        'Arrival_min': 0,
        'Total_Duration_mins': 240,
        'Route_Stops': 0,
        'Route_Source_Match': 1,
        'Route_Dest_Match': 1,
        'Airline': 'IndiGo',
        'Source': 'Delhi',
        'Destination': 'Mumbai',
        'Route_Complexity': 'Direct'
    }
    
    query_lower = query.lower()
    
    # Extract source and destination
    from_match = re.search(r'from (\w+)', query_lower)
    to_match = re.search(r'to (\w+)', query_lower)
    
    if from_match:
        features['Source'] = from_match.group(1).title()
    if to_match:
        features['Destination'] = to_match.group(1).title()
    
    # Extract airline
    airlines = ['indigo', 'air india', 'spicejet', 'vistara', 'goair', 'jet airways', 'alliance air']
    for airline in airlines:
        if airline in query_lower:
            features['Airline'] = airline.title()
            break
    
    # Extract month
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    for month_name, month_num in months.items():
        if month_name in query_lower:
            features['Journey_month'] = month_num
            break
    
    # Extract stops
    if 'non-stop' in query_lower or 'direct' in query_lower:
        features['Total_Stops'] = 0
    elif '1 stop' in query_lower or 'one stop' in query_lower:
        features['Total_Stops'] = 1
    elif '2 stop' in query_lower or 'two stop' in query_lower:
        features['Total_Stops'] = 2
    
    return features

def handle_ml_query(query):
    """
    Handle ML flight price prediction queries
    """
    if not ml_loaded:
        return {
            "answer": "ML model is not available. Please train and save the flight price prediction model first.",
            "error": True
        }
    
    try:
        # Extract features from query
        features = extract_features_from_query(query)
        
        # Debug: Print features
        print(f"Extracted features: {features}")
        
        # Create DataFrame for prediction
        df = pd.DataFrame([features])
        
        # Debug: Print DataFrame info
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame dtypes:\n{df.dtypes}")
        
        # Make prediction
        prediction = ml_pipeline.predict(df)
        
        # Debug: Print raw prediction
        print(f"Raw prediction: {prediction}")
        print(f"Prediction type: {type(prediction)}")
        print(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'No shape'}")
        
        # Handle different prediction formats
        if isinstance(prediction, (list, np.ndarray)):
            if len(prediction) > 0:
                pred_value = prediction[0]
            else:
                raise ValueError("Empty prediction array")
        else:
            pred_value = prediction
        
        # Convert to float and validate
        try:
            pred_float = float(pred_value)
            if pred_float <= 0 or pred_float > 1000000:  # Reasonable price range
                raise ValueError(f"Unreasonable price prediction: {pred_float}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid prediction value: {pred_value}, type: {type(pred_value)}")
        
        return {
            "answer": f"🛫 **Flight Price Prediction**\n\n💰 **Estimated Price: ₹{pred_float:,.2f}**\n\n📋 **Based on:**\n- **Route:** {features.get('Source', 'Unknown')} → {features.get('Destination', 'Unknown')}\n- **Airline:** {features.get('Airline', 'Unknown')}\n- **Month:** {features.get('Journey_month', 'Unknown')}\n- **Stops:** {features.get('Total_Stops', 'Unknown')}\n\n*This is an estimated price based on historical data.*",
            "predicted_price": pred_float,
            "features": features,
            "debug_info": {
                "raw_prediction": str(prediction),
                "prediction_type": str(type(prediction)),
                "feature_count": len(features)
            }
        }
        
    except Exception as e:
        # Fallback to simple rule-based estimation
        try:
            fallback_price = simple_price_estimate(features)
            return {
                "answer": f"🛫 **Flight Price Prediction (Estimated)**\n\n💰 **Estimated Price: ₹{fallback_price:,.2f}**\n\n*This is a rough estimate based on available data.*",
                "predicted_price": fallback_price,
                "features": features,
                "debug_info": {
                    "error": str(e),
                    "fallback_used": True
                }
            }
        except Exception as e2:
            return {
                "answer": f"Sorry, I couldn't predict the flight price. Error: {str(e)}",
                "error": True,
                "debug_info": {
                    "error_type": type(e).__name__,
                    "features": features if 'features' in locals() else "Not extracted"
                }
            }

def simple_price_estimate(features):
    """
    Simple rule-based price estimation when ML model fails
    """
    # Base price
    base_price = 5000
    
    # Route-based adjustments
    route_multipliers = {
        ('Delhi', 'Mumbai'): 1.0,
        ('Mumbai', 'Delhi'): 1.0,
        ('Bangalore', 'Delhi'): 1.2,
        ('Delhi', 'Bangalore'): 1.2,
        ('Chennai', 'Mumbai'): 0.9,
        ('Mumbai', 'Chennai'): 0.9,
        ('Kolkata', 'Delhi'): 1.1,
        ('Delhi', 'Kolkata'): 1.1
    }
    
    source = features.get('Source', 'Unknown')
    destination = features.get('Destination', 'Unknown')
    route_key = (source, destination)
    
    price = base_price * route_multipliers.get(route_key, 1.0)
    
    # Airline adjustments
    airline_multipliers = {
        'IndiGo': 1.0,
        'Air India': 1.1,
        'SpiceJet': 0.9,
        'Vistara': 1.3,
        'GoAir': 0.85,
        'Jet Airways': 1.2,
        'Alliance Air': 0.8
    }
    
    airline = features.get('Airline', 'IndiGo')
    price *= airline_multipliers.get(airline, 1.0)
    
    # Stops adjustment
    stops = features.get('Total_Stops', 0)
    if stops == 0:
        price *= 1.0  # Direct flight
    elif stops == 1:
        price *= 0.8  # One stop cheaper
    else:
        price *= 0.7  # Multiple stops cheapest
    
    # Month adjustment (peak/off-peak)
    month = features.get('Journey_month', 6)
    peak_months = [12, 1, 4, 5, 10, 11]  # Winter and festival months
    if month in peak_months:
        price *= 1.2
    else:
        price *= 0.9
    
    return round(price, 2)

# ==== ENDPOINTS ====
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Route the query
    route = route_query(query)
    
    if route == "ML":
        result = handle_ml_query(query)
    else:
        result = handle_rag_query(query)

    return jsonify({
        "query": query,
        "route": route,
        "result": result
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "ml_model_loaded": ml_loaded,
        "rag_loaded": rag_loaded,
        "intent_detection": "keyword_based"
    })

@app.route("/test_ml", methods=["GET"])
def test_ml():
    """Test ML model with sample data"""
    if not ml_loaded:
        return jsonify({
            "error": "ML model not loaded",
            "ml_pipeline": str(type(ml_pipeline)),
            "feature_names": str(type(feature_names))
        })
    
    try:
        # Create sample features
        sample_features = {
            'Total_Stops': 0,
            'Journey_day': 15,
            'Journey_month': 6,
            'Journey_year': 2024,
            'Dep_hour': 10,
            'Dep_min': 0,
            'Arrival_hour': 14,
            'Arrival_min': 0,
            'Total_Duration_mins': 240,
            'Route_Stops': 0,
            'Route_Source_Match': 1,
            'Route_Dest_Match': 1,
            'Airline': 'IndiGo',
            'Source': 'Delhi',
            'Destination': 'Mumbai',
            'Route_Complexity': 'Direct'
        }
        
        df = pd.DataFrame([sample_features])
        prediction = ml_pipeline.predict(df)
        
        return jsonify({
            "status": "success",
            "sample_features": sample_features,
            "dataframe_shape": df.shape,
            "dataframe_columns": df.columns.tolist(),
            "raw_prediction": str(prediction),
            "prediction_type": str(type(prediction)),
            "prediction_value": float(prediction[0]) if len(prediction) > 0 else "No prediction"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "ml_pipeline_type": str(type(ml_pipeline)),
            "feature_names_type": str(type(feature_names))
        })

# ==== RUN ====
if __name__ == "__main__":
    app.run(debug=True, port=5000)
