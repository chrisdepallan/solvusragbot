# Smart Travel Advisor - Web Interface

A modern chatbot web interface for the Smart Travel Advisor system that provides:
- ✈️ Flight price predictions using ML models
- 📋 Travel policy information using RAG (Retrieval-Augmented Generation)
- 🤖 Intelligent intent detection with OpenAI
- 💬 Beautiful, responsive chat interface

## Features

### 🎯 Intent Detection
- Automatically routes queries to appropriate systems (RAG or ML)
- Uses OpenAI for intelligent query analysis
- Fallback to keyword matching for reliability

### 🤖 RAG System
- Answers questions about travel policies, procedures, and documentation
- Uses FAISS for efficient vector search
- Provides context-aware responses with source references

### 📊 ML Predictions
- Predicts flight prices based on route, airline, and travel details
- Trained Random Forest/XGBoost models
- Extracts travel details from natural language queries

### 🎨 Modern UI
- Responsive design that works on all devices
- Real-time typing indicators and status updates
- Quick action buttons for common queries
- Toast notifications for system feedback
- Export chat history functionality

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API=your_openai_api_key_here
```

### 3. Required Files
Make sure these files exist in your project directory:
- `metadata.pkl` - Document metadata for RAG
- `all_chunks.pkl` - Document chunks for RAG (optional, will create dummy data)
- `faiss_index.index` - FAISS vector index
- `flight_price_prediction_pipeline.joblib` - ML model pipeline (optional)
- `feature_names.joblib` - ML feature names (optional)

### 4. Run the Application

#### Option A: Run with ML Model (Full Features)
```bash
python app.py
```

#### Option B: Run RAG Only (Simplified)
```bash
python app_web.py
```

#### Option C: Use the Batch File (Windows)
```bash
start_web.bat
```

### 5. Access the Interface
Open your browser and navigate to:
- Main Interface: http://localhost:5000
- Test Interface: http://localhost:5000/test
- Health Check: http://localhost:5000/health

## Usage Examples

### RAG Queries (Policy & Information)
- "What is the travel policy for international flights?"
- "How do I submit expense reports?"
- "What documents are required for business travel?"
- "Can I book flights for my family?"

### ML Queries (Price Predictions)
- "Predict flight price from Delhi to Mumbai"
- "How much does a flight from Bangalore to New Delhi cost?"
- "Estimate airfare for IndiGo flight from BOM to DEL"
- "Flight price prediction for next month"

## File Structure

```
solvusragbot/
├── app.py                    # Main Flask application (full features)
├── app_web.py               # Simplified Flask application (RAG only)
├── templates/
│   ├── index.html           # Main chat interface
│   └── test.html            # Simple test interface
├── static/
│   ├── css/
│   │   └── style.css        # Styles for the chat interface
│   └── js/
│       └── script.js        # JavaScript for chat functionality
├── requirements.txt         # Python dependencies
├── start_web.bat           # Windows startup script
└── .env                    # Environment variables
```

## API Endpoints

### POST /ask
Send a query to the chatbot:
```json
{
  "query": "What is the travel policy?"
}
```

Response:
```json
{
  "query": "What is the travel policy?",
  "route": "RAG",
  "intent_info": {
    "route": "RAG",
    "reason": "Travel policy inquiry",
    "query_type": "policy_question"
  },
  "result": {
    "answer": "The travel policy states...",
    "query_type": "policy_question",
    "matched_documents": ["doc1", "doc2"],
    "intent_reason": "Travel policy inquiry"
  }
}
```

### GET /health
Check system status:
```json
{
  "status": "healthy",
  "ml_model_loaded": true,
  "rag_loaded": true,
  "intent_detection": "openai_function_calling"
}
```

## Troubleshooting

### Fixed Issues ✅
The following issues have been resolved in the current version:

1. **AttributeError: 'function' object has no attribute 'search'** - Fixed by properly naming FAISS index variable
2. **ML Model Loading Issues** - Added proper error handling and graceful degradation
3. **Intent Detection Complexity** - Simplified to keyword-based routing for reliability

### Common Issues

1. **Missing OpenAI API Key**
   - Make sure `.env` file exists with `OPENAI_API=your_key`
   - Check that the key is valid and has sufficient credits

2. **Missing Required Files**
   - Ensure `metadata.pkl` and `faiss_index.index` exist
   - `all_chunks.pkl` is optional (will create dummy data if missing)

3. **ML Model Not Loading**
   - Check if `flight_price_prediction_pipeline.joblib` exists
   - Verify the model was trained and saved correctly
   - App will work in RAG-only mode if ML model is missing

4. **FAISS Index Issues**
   - Rebuild the FAISS index if corrupted
   - Check that the index dimensions match your embedding model

5. **Port Already in Use**
   - Change the port in `app.py`: `app.run(debug=True, port=5001)`
   - Or kill the existing process

### Development Mode

To run in development mode with auto-reload:
```bash
set FLASK_ENV=development
set FLASK_APP=app.py
flask run
```

## Customization

### Styling
- Modify `static/css/style.css` to change the appearance
- Update colors, fonts, and layout as needed

### Functionality
- Add new quick action buttons in `templates/index.html`
- Modify intent detection logic in the Flask app
- Add new API endpoints for additional features

### UI Components
- Update `static/js/script.js` for new chat features
- Add new message types and formatting options
- Implement additional user interactions

## Performance Tips

1. **Caching**: Implement response caching for common queries
2. **Async Processing**: Use background tasks for heavy ML computations
3. **CDN**: Host static assets on a CDN for better performance
4. **Database**: Store chat history in a database for persistence

## Security Considerations

1. **API Key Protection**: Never expose OpenAI API keys in client-side code
2. **Input Validation**: Validate and sanitize all user inputs
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **HTTPS**: Use HTTPS in production environments
