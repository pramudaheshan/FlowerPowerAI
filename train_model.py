"""
Train and save the Iris flower classification model
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_iris_model():
    """Train and save the Iris classification model"""
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Save the model
    joblib.dump(model, "model.pkl")
    print("\nModel saved as 'model.pkl'")
    
    return model, iris.target_names

if __name__ == "__main__":
    train_iris_model()
