"""
Sentiment Analysis API Backend
Flask server that loads trained models and provides prediction endpoints
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import re
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and vectorizer
model = None
vectorizer = None

def clean_text(text):
    """Simple text cleaning for prediction"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_models():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models'))
    
    # Try to load saved models
    try:
        model = joblib.load(os.path.join(model_path, 'lr_clothing.pkl'))
        vectorizer = joblib.load(os.path.join(model_path, 'tfidf_clothing.pkl'))
        print("✓ Loaded saved models from '../saved_models/' directory")
        return True
    except FileNotFoundError:
        print("⚠ No saved models found. Please run the notebook first to save models.")
        print("  Expected files:")
        print("  - ../saved_models/lr_clothing.pkl")
        print("  - ../saved_models/tfidf_clothing.pkl")
        return False

# HTML Template with embedded CSS and JS
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 600px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        
        .header h1 {
            color: #fff;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.95rem;
        }
        
        .input-group {
            margin-bottom: 24px;
        }
        
        .input-group label {
            display: block;
            color: rgba(255, 255, 255, 0.8);
            font-weight: 500;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }
        
        textarea {
            width: 100%;
            min-height: 150px;
            padding: 16px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            resize: vertical;
            transition: all 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
        }
        
        textarea::placeholder {
            color: rgba(255, 255, 255, 0.4);
        }
        
        .btn {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px -10px rgba(102, 126, 234, 0.5);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 32px;
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            display: none;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result.positive {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2));
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .result.neutral {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.2));
            border: 1px solid rgba(251, 191, 36, 0.3);
        }
        
        .result.negative {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2));
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .result-icon {
            font-size: 3rem;
            margin-bottom: 12px;
        }
        
        .result-label {
            font-size: 1.5rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 8px;
        }
        
        .result-confidence {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 12px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .footer {
            text-align: center;
            margin-top: 24px;
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <h1>🎯 Sentiment Analyzer</h1>
                <p>Enter a product review to analyze its sentiment</p>
            </div>
            
            <div class="input-group">
                <label for="review">Your Review</label>
                <textarea 
                    id="review" 
                    placeholder="Type or paste a product review here...

Example: This product exceeded my expectations! The quality is amazing and it arrived quickly."
                ></textarea>
            </div>
            
            <button class="btn" id="analyzeBtn" onclick="analyze()">
                Analyze Sentiment
            </button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="color: rgba(255,255,255,0.6);">Analyzing...</p>
            </div>
            
            <div class="result" id="result">
                <div class="result-icon" id="resultIcon"></div>
                <div class="result-label" id="resultLabel"></div>
                <div class="result-confidence" id="resultConfidence"></div>
            </div>
            
            <div class="footer">
                Powered by Machine Learning • TF-IDF + Logistic Regression
            </div>
        </div>
    </div>
    
    <script>
        async function analyze() {
            const review = document.getElementById('review').value.trim();
            
            if (!review) {
                alert('Please enter a review to analyze.');
                return;
            }
            
            const btn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            // Show loading
            btn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ review: review })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Show result
                const icons = { 'Positive': '😊', 'Neutral': '😐', 'Negative': '😞' };
                const classes = { 'Positive': 'positive', 'Neutral': 'neutral', 'Negative': 'negative' };
                
                document.getElementById('resultIcon').textContent = icons[data.sentiment];
                document.getElementById('resultLabel').textContent = data.sentiment;
                document.getElementById('resultConfidence').textContent = 
                    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                
                result.className = 'result ' + classes[data.sentiment];
                result.style.display = 'block';
                
            } catch (error) {
                alert('Error connecting to server. Make sure the Flask server is running.');
                console.error(error);
            } finally {
                btn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        // Allow Enter key to submit (Ctrl+Enter for newlines)
        document.getElementById('review').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                analyze();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Serve the frontend"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for a given review"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please run the notebook first to save models.'}), 500
    
    try:
        data = request.get_json()
        review = data.get('review', '')
        
        if not review:
            return jsonify({'error': 'No review provided'}), 400
        
        # Clean and vectorize
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        # Map prediction to label
        labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = labels.get(prediction, 'Unknown')
        confidence = float(max(probabilities))
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'Negative': float(probabilities[0]),
                'Neutral': float(probabilities[1]),
                'Positive': float(probabilities[2])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎯 Sentiment Analysis Server")
    print("="*50)
    
    # Try to load models
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n⚠ Running in demo mode (no model loaded)")
        print("  To use the model, run the notebook and save models first.")
    
    print("\n🌐 Starting server at http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
