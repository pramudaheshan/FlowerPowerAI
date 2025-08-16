"""
FastAPI application for Iris flower classification
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="A machine learning API to classify Iris flowers based on their measurements",
    version="1.0.0"
)

# Load the trained model at startup
try:
    model = joblib.load("model.pkl")
    class_names = ["setosa", "versicolor", "virginica"]
except FileNotFoundError:
    raise RuntimeError("Model file 'model.pkl' not found. Please train the model first.")

# Pydantic models for request and response
class IrisInput(BaseModel):
    """Input data model for Iris flower measurements"""
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=10)
    
    @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    @classmethod
    def validate_measurements(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Measurements must be between 0 and 10 cm')
        return v

class PredictionOutput(BaseModel):
    """Output data model for prediction results"""
    species: str = Field(..., description="Predicted Iris species")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: dict = Field(..., description="Probabilities for each class")

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    is_model_loaded: bool = Field(..., description="Whether the model is loaded")

# API Endpoints
@app.get("/", response_class=HTMLResponse, summary="Interactive Web Interface")
async def root():
    """Interactive web interface for testing the Iris classification API"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🌸 Iris Flower Classification</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .content {
                padding: 40px;
            }
            
            .form-section {
                margin-bottom: 30px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                font-weight: bold;
                color: #333;
                margin-bottom: 8px;
                font-size: 1.1em;
            }
            
            .form-group input {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e1e1e1;
                border-radius: 10px;
                font-size: 1em;
                transition: all 0.3s ease;
            }
            
            .form-group input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .input-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 20px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .result {
                margin-top: 30px;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid;
                display: none;
            }
            
            .result.success {
                background: #d4edda;
                border-color: #28a745;
                color: #155724;
            }
            
            .result.error {
                background: #f8d7da;
                border-color: #dc3545;
                color: #721c24;
            }
            
            .species-result {
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 15px;
            }
            
            .confidence-bar {
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .confidence-fill {
                background: linear-gradient(90deg, #28a745, #20c997);
                height: 25px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                transition: width 0.5s ease;
            }
            
            .probabilities {
                margin-top: 20px;
            }
            
            .prob-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 8px 0;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            
            .species-icons {
                text-align: center;
                margin: 20px 0;
                font-size: 3em;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .examples {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
                margin-top: 30px;
            }
            
            .example-btn {
                background: #6c757d;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                margin: 5px;
                cursor: pointer;
                font-size: 0.9em;
            }
            
            .example-btn:hover {
                background: #5a6268;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌸 Iris Flower Classifier</h1>
                <p>AI-powered species identification using machine learning</p>
            </div>
            
            <div class="content">
                <div class="form-section">
                    <h2>Enter Flower Measurements</h2>
                    <p style="color: #666; margin-bottom: 20px;">Provide the measurements in centimeters (0-10 cm)</p>
                    
                    <form id="irisForm">
                        <div class="input-grid">
                            <div class="form-group">
                                <label for="sepal_length">🌿 Sepal Length (cm)</label>
                                <input type="number" id="sepal_length" step="0.1" min="0" max="10" required>
                            </div>
                            <div class="form-group">
                                <label for="sepal_width">🌿 Sepal Width (cm)</label>
                                <input type="number" id="sepal_width" step="0.1" min="0" max="10" required>
                            </div>
                            <div class="form-group">
                                <label for="petal_length">🌺 Petal Length (cm)</label>
                                <input type="number" id="petal_length" step="0.1" min="0" max="10" required>
                            </div>
                            <div class="form-group">
                                <label for="petal_width">🌺 Petal Width (cm)</label>
                                <input type="number" id="petal_width" step="0.1" min="0" max="10" required>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn" id="predictBtn">
                            🔍 Classify Flower
                        </button>
                    </form>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing flower characteristics...</p>
                </div>
                
                <div class="result" id="result">
                    <div class="species-icons" id="speciesIcons"></div>
                    <div class="species-result" id="speciesResult"></div>
                    <div>
                        <strong>Confidence:</strong>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidenceFill">0%</div>
                        </div>
                    </div>
                    <div class="probabilities" id="probabilities"></div>
                </div>
                
                <div class="examples">
                    <h3>📋 Try These Examples</h3>
                    <p style="margin-bottom: 15px;">Click to auto-fill the form with sample data:</p>
                    <button class="example-btn" onclick="fillExample(5.1, 3.5, 1.4, 0.2)">🌸 Setosa Example</button>
                    <button class="example-btn" onclick="fillExample(7.0, 3.2, 4.7, 1.4)">🌼 Versicolor Example</button>
                    <button class="example-btn" onclick="fillExample(6.3, 3.3, 6.0, 2.5)">🌺 Virginica Example</button>
                </div>
            </div>
        </div>
        
        <script>
            const speciesIcons = {
                'setosa': '🌸',
                'versicolor': '🌼', 
                'virginica': '🌺'
            };
            
            const speciesColors = {
                'setosa': '#ff6b6b',
                'versicolor': '#4ecdc4',
                'virginica': '#45b7d1'
            };
            
            function fillExample(sl, sw, pl, pw) {
                document.getElementById('sepal_length').value = sl;
                document.getElementById('sepal_width').value = sw;
                document.getElementById('petal_length').value = pl;
                document.getElementById('petal_width').value = pw;
            }
            
            document.getElementById('irisForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value)
                };
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('predictBtn').disabled = true;
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayResult(data);
                    } else {
                        displayError(data.detail || 'An error occurred');
                    }
                } catch (error) {
                    displayError('Network error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('predictBtn').disabled = false;
                }
            });
            
            function displayResult(data) {
                const resultDiv = document.getElementById('result');
                const speciesResult = document.getElementById('speciesResult');
                const confidenceFill = document.getElementById('confidenceFill');
                const probabilitiesDiv = document.getElementById('probabilities');
                const speciesIconsDiv = document.getElementById('speciesIcons');
                
                // Set species result
                const species = data.species;
                const confidence = (data.confidence * 100).toFixed(1);
                
                speciesResult.textContent = `Species: ${species.charAt(0).toUpperCase() + species.slice(1)}`;
                speciesResult.style.color = speciesColors[species];
                
                // Set icons
                speciesIconsDiv.textContent = speciesIcons[species].repeat(3);
                
                // Set confidence bar
                confidenceFill.style.width = confidence + '%';
                confidenceFill.textContent = confidence + '%';
                
                // Set probabilities
                probabilitiesDiv.innerHTML = '<h4>All Probabilities:</h4>';
                for (const [spec, prob] of Object.entries(data.probabilities)) {
                    const probPercent = (prob * 100).toFixed(1);
                    const probItem = document.createElement('div');
                    probItem.className = 'prob-item';
                    probItem.innerHTML = `
                        <span>${speciesIcons[spec]} ${spec.charAt(0).toUpperCase() + spec.slice(1)}</span>
                        <span><strong>${probPercent}%</strong></span>
                    `;
                    probabilitiesDiv.appendChild(probItem);
                }
                
                resultDiv.className = 'result success';
                resultDiv.style.display = 'block';
            }
            
            function displayError(message) {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `<h3>❌ Error</h3><p>${message}</p>`;
                resultDiv.className = 'result error';
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthCheck, summary="Health check")
async def health_check():
    """Check if the API and model are working properly"""
    return HealthCheck(
        status="healthy",
        is_model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionOutput, summary="Predict Iris species")
async def predict_iris(input_data: IrisInput):
    """
    Predict the Iris flower species based on sepal and petal measurements.
    
    - **sepal_length**: Length of the sepal in cm (0-10)
    - **sepal_width**: Width of the sepal in cm (0-10)
    - **petal_length**: Length of the petal in cm (0-10)
    - **petal_width**: Width of the petal in cm (0-10)
    
    Returns the predicted species with confidence score and all class probabilities.
    """
    try:
        # Prepare the input features
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Prepare response
        predicted_species = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        # Create probabilities dictionary
        prob_dict = {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
        
        return PredictionOutput(
            species=predicted_species,
            confidence=confidence,
            probabilities=prob_dict
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/batch", summary="Batch prediction")
async def predict_batch(input_list: List[IrisInput]):
    """
    Predict multiple Iris flowers at once.
    
    Takes a list of flower measurements and returns predictions for each.
    """
    try:
        predictions = []
        
        for input_data in input_list:
            # Prepare features
            features = np.array([[
                input_data.sepal_length,
                input_data.sepal_width,
                input_data.petal_length,
                input_data.petal_width
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Prepare response
            predicted_species = class_names[prediction]
            confidence = float(probabilities[prediction])
            
            prob_dict = {
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            }
            
            predictions.append(PredictionOutput(
                species=predicted_species,
                confidence=confidence,
                probabilities=prob_dict
            ))
        
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
