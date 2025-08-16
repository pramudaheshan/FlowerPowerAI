"""
Test examples for the Iris Classification API
"""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8001"

# Test data examples
test_examples = [
    {
        "name": "Setosa Example",
        "data": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "expected": "setosa"
    },
    {
        "name": "Versicolor Example",
        "data": {
            "sepal_length": 7.0,
            "sepal_width": 3.2,
            "petal_length": 4.7,
            "petal_width": 1.4
        },
        "expected": "versicolor"
    },
    {
        "name": "Virginica Example",
        "data": {
            "sepal_length": 6.3,
            "sepal_width": 3.3,
            "petal_length": 6.0,
            "petal_width": 2.5
        },
        "expected": "virginica"
    }
]

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with example data"""
    print("\n" + "="*50)
    print("TESTING PREDICTION ENDPOINT")
    print("="*50)
    
    for example in test_examples:
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=example["data"]
            )
            
            print(f"\n{example['name']}:")
            print(f"Input: {example['data']}")
            print(f"Expected: {example['expected']}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Predicted: {result['species']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"All Probabilities: {result['probabilities']}")
                
                # Check if prediction matches expected
                if result['species'] == example['expected']:
                    print("✅ Prediction CORRECT")
                else:
                    print("❌ Prediction INCORRECT")
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Test failed: {e}")

def test_batch_prediction():
    """Test the batch prediction endpoint"""
    print("\n" + "="*50)
    print("TESTING BATCH PREDICTION ENDPOINT")
    print("="*50)
    
    batch_data = [example["data"] for example in test_examples]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Batch predictions for {len(batch_data)} samples:")
            
            for i, prediction in enumerate(result["predictions"]):
                print(f"\nSample {i+1}:")
                print(f"  Species: {prediction['species']}")
                print(f"  Confidence: {prediction['confidence']:.4f}")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Batch test failed: {e}")

def test_invalid_input():
    """Test error handling with invalid input"""
    print("\n" + "="*50)
    print("TESTING ERROR HANDLING")
    print("="*50)
    
    invalid_examples = [
        {
            "name": "Negative values",
            "data": {
                "sepal_length": -1.0,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        },
        {
            "name": "Values too large",
            "data": {
                "sepal_length": 15.0,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        },
        {
            "name": "Missing field",
            "data": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4
                # missing petal_width
            }
        }
    ]
    
    for example in invalid_examples:
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=example["data"]
            )
            
            print(f"\n{example['name']}:")
            print(f"Input: {example['data']}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 422:  # Validation error
                print("✅ Validation error correctly handled")
            else:
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Test failed: {e}")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API is running on http://localhost:8001")
    
    # Test health endpoint
    if test_health_endpoint():
        print("✅ Health check passed")
        
        # Test prediction endpoints
        test_prediction_endpoint()
        test_batch_prediction()
        test_invalid_input()
        
        print("\n" + "="*50)
        print("All tests completed!")
        print("="*50)
    else:
        print("❌ Health check failed. Is the API running?")
