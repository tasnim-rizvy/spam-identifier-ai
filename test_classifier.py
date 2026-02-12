import joblib
import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def check_files_exist():
    print("=" * 60)
    print("🔍 CHECKING FILES")
    print("=" * 60)
    
    required_files = {
        "Model": "models/spam_model.pkl",
        "Vectorizer": "models/vectorizer.pkl",
        "Data": "data/SMSSpamCollection"
    }
    
    all_exist = True
    for name, path in required_files.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} NOT FOUND!")
            all_exist = False
    
    print()
    return all_exist

def load_test_data():
    spam_examples = [
        "WINNER!! You have won a £1000 cash prize! Call 09061701461 to claim.",
        "Congratulations! You've been selected for a FREE iPhone. Click here now!",
        "URGENT! Your account has been compromised. Reply with your password immediately.",
        "Get paid to work from home! Make $5000 per week! No experience needed!",
        "Claim your FREE prize now! Limited time offer! Text PRIZE to 12345"
    ]
    
    ham_examples = [
        "Hey, are we still meeting for lunch at 1pm?",
        "Can you pick up some milk on your way home?",
        "Thanks for the birthday wishes! Had a great day.",
        "Meeting rescheduled to Thursday at 3pm. See you then!",
        "I'll be there in 10 minutes. Traffic is pretty bad."
    ]
    
    return spam_examples, ham_examples

def test_basic_functionality():
    print("=" * 60)
    print("🧪 TEST 1: Basic Functionality")
    print("=" * 60)
    
    try:
        model = joblib.load("models/spam_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        print("✅ Models loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load models: {e}\n")
        return False
    
    spam_examples, ham_examples = load_test_data()
    
    print("Testing SPAM messages:")
    print("-" * 60)
    spam_correct = 0
    for msg in spam_examples:
        vec = vectorizer.transform([msg])
        prob = model.predict_proba(vec)[0][1]
        prediction = "SPAM" if prob >= 0.3 else "HAM"
        status = "✅" if prediction == "SPAM" else "❌"
        print(f"{status} {prediction} ({prob:.2%}) - {msg[:50]}...")
        if prediction == "SPAM":
            spam_correct += 1
    
    print(f"\nSpam Detection Rate: {spam_correct}/{len(spam_examples)} ({spam_correct/len(spam_examples)*100:.0f}%)\n")
    
    print("Testing HAM messages:")
    print("-" * 60)
    ham_correct = 0
    for msg in ham_examples:
        vec = vectorizer.transform([msg])
        prob = model.predict_proba(vec)[0][1]
        prediction = "SPAM" if prob >= 0.3 else "HAM"
        status = "✅" if prediction == "HAM" else "❌"
        print(f"{status} {prediction} ({prob:.2%}) - {msg[:50]}...")
        if prediction == "HAM":
            ham_correct += 1
    
    print(f"\nHam Detection Rate: {ham_correct}/{len(ham_examples)} ({ham_correct/len(ham_examples)*100:.0f}%)\n")
    
    total_correct = spam_correct + ham_correct
    total_tests = len(spam_examples) + len(ham_examples)
    print(f"Overall Accuracy: {total_correct}/{total_tests} ({total_correct/total_tests*100:.0f}%)\n")
    
    return True

def test_edge_cases():
    print("=" * 60)
    print("🧪 TEST 2: Edge Cases")
    print("=" * 60)
    
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    
    edge_cases = [
        ("Empty-ish", "   "),
        ("Single word", "hello"),
        ("Numbers only", "123456"),
        ("Special chars", "!@#$%^&*()"),
        ("Very long", "hello " * 100),
        ("Mixed case", "HeLLo WoRLd"),
        ("Emojis/Unicode", "Hello 😊 world 🎉")
    ]
    
    print("Testing edge cases:")
    print("-" * 60)
    
    for name, msg in edge_cases:
        try:
            vec = vectorizer.transform([msg])
            prob = model.predict_proba(vec)[0][1]
            print(f"✅ {name:15} - Probability: {prob:.4f}")
        except Exception as e:
            print(f"❌ {name:15} - Error: {e}")
    
    print()

def test_threshold_sensitivity():
    print("=" * 60)
    print("🧪 TEST 3: Threshold Sensitivity")
    print("=" * 60)
    
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    
    test_msg = "Congratulations! You won a prize. Call now!"
    vec = vectorizer.transform([test_msg])
    prob = model.predict_proba(vec)[0][1]
    
    print(f"Test Message: '{test_msg}'")
    print(f"Spam Probability: {prob:.4f}\n")
    
    print("Classification at different thresholds:")
    print("-" * 60)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        prediction = "SPAM" if prob >= threshold else "HAM"
        print(f"Threshold {threshold:.1f}: {prediction}")
    
    print()

def analyze_model_characteristics():
    print("=" * 60)
    print("🧪 TEST 4: Model Characteristics")
    print("=" * 60)
    
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    print(f"Total Features: {len(feature_names)}")
    print(f"Alpha (smoothing): {model.alpha}")
    print(f"Number of classes: {len(model.classes_)}")
    print(f"Classes: {model.classes_}\n")
    
    # Get top features for spam
    log_probs = model.feature_log_prob_
    
    print("🔴 Top 15 SPAM indicators:")
    print("-" * 60)
    spam_indices = np.argsort(log_probs[1])[-15:][::-1]
    for i, idx in enumerate(spam_indices, 1):
        print(f"{i:2}. {feature_names[idx]}")
    
    print("\n🟢 Top 15 HAM indicators:")
    print("-" * 60)
    ham_indices = np.argsort(log_probs[0])[-15:][::-1]
    for i, idx in enumerate(ham_indices, 1):
        print(f"{i:2}. {feature_names[idx]}")
    
    print()

def test_custom_messages():
    print("=" * 60)
    print("🧪 TEST 5: Test Your Own Messages")
    print("=" * 60)
    
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    
    print("Enter messages to test (type 'done' to finish):\n")
    
    while True:
        msg = input("📧 Message: ").strip()
        
        if msg.lower() == 'done':
            break
        
        if not msg:
            continue
        
        try:
            vec = vectorizer.transform([msg])
            prob = model.predict_proba(vec)[0][1]
            prediction = "SPAM" if prob >= 0.3 else "HAM"
            
            print(f"   Result: {prediction} (confidence: {prob:.2%})\n")
        except Exception as e:
            print(f"   Error: {e}\n")

def main():
    print("\n" + "=" * 60)
    print("🚀 SPAM CLASSIFIER TEST SUITE")
    print("=" * 60 + "\n")
    
    # Check files
    if not check_files_exist():
        print("❌ Some required files are missing!")
        print("Please run train.py first to create the model.\n")
        return
    
    try:
        # Run tests
        test_basic_functionality()
        test_edge_cases()
        test_threshold_sensitivity()
        analyze_model_characteristics()
        
        # Interactive testing
        response = input("Would you like to test your own messages? (y/n): ").strip().lower()
        if response == 'y':
            test_custom_messages()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}\n")

if __name__ == "__main__":
    main()