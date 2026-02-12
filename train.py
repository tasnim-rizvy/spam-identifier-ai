from sklearn.model_selection import train_test_split
from src.features.feature_extraction import extract_features
from src.model import SpamClassifier
import joblib
import os
import sys

def split_data():
    print("📊 Loading and extracting features...")
    
    try:
        X, y, vectorizer = extract_features()
        print(f"✅ Loaded {len(y)} messages")
        print(f"   - Spam messages: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"   - Ham messages: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        
    except FileNotFoundError:
        print("❌ Error: Data file not found!")
        print("Please make sure 'data/SMSSpamCollection' exists.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y 
    )
    
    print(f"✅ Split into {X_train.shape[0]} training and {X_test.shape[0]} test samples\n")
    
    return X_train, X_test, y_train, y_test, vectorizer

def train_model(X_train, y_train, alpha=0.5):
    print("🤖 Training the model...")
    
    classifier = SpamClassifier(alpha=alpha)
    classifier.train(X_train, y_train)
    
    print("✅ Model trained successfully!\n")
    
    return classifier

def save_models(classifier, vectorizer):
    print("💾 Saving models...")
    
    os.makedirs("models", exist_ok=True)
    
    try:
        joblib.dump(classifier.model, "models/spam_model.pkl")
        joblib.dump(vectorizer, "models/vectorizer.pkl")
        print("✅ Models saved to 'models/' directory\n")
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        sys.exit(1)

def evaluate_model(classifier, X_test, y_test):
    print("=" * 60)
    print("📈 MODEL EVALUATION")
    print("=" * 60)
    
    thresholds = [0.5, 0.4, 0.3, 0.2]
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        print(f"\n{'=' * 60}")
        print(f"Threshold: {threshold}")
        print(f"{'=' * 60}")
        
        metrics = classifier.evaluate(X_test, y_test, threshold=threshold)
        
        print(f"\n📊 Performance Metrics:")
        print(f"  • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  • Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  • Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  • F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        
        print(f"\n📉 Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Ham    Spam")
        print(f"Actual Ham    {metrics['confusion_matrix'][0][0]:<6} {metrics['confusion_matrix'][0][1]:<6}")
        print(f"       Spam   {metrics['confusion_matrix'][1][0]:<6} {metrics['confusion_matrix'][1][1]:<6}")
        
        print(f"\n📋 Classification Report:")
        print(metrics['classification_report'])
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
    
    print(f"\n{'=' * 60}")
    print(f"🏆 Best Threshold: {best_threshold} (F1 Score: {best_f1:.4f})")
    print(f"{'=' * 60}\n")

def main():
    print("\n" + "=" * 60)
    print("🚀 SPAM CLASSIFIER TRAINING")
    print("=" * 60 + "\n")
    
    # Step 1: Load and split data
    X_train, X_test, y_train, y_test, vectorizer = split_data()
    
    # Step 2: Train model
    classifier = train_model(X_train, y_train, alpha=0.5)
    
    # Step 3: Save models
    save_models(classifier, vectorizer)
    
    # Step 4: Evaluate model
    evaluate_model(classifier, X_test, y_test)
    
    print("✅ Training complete! You can now run predict.py")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()