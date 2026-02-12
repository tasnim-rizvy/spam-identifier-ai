import joblib
import os
import sys

def load_models():
    try:
        if not os.path.exists("models/spam_model.pkl"):
            print("❌ Error: Model file not found!")
            print("Please run train.py first to create the model.")
            sys.exit(1)

        if not os.path.exists("models/vectorizer.pkl"):
            print("❌ Error: Vectorizer file not found!")
            print("Please run train.py first to create the vectorizer.")
            sys.exit(1)

        model = joblib.load("models/spam_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")

        print("✅ Models loaded successfully!\n")
        return model, vectorizer
    
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)



def predict_message(message, model, vectorizer, threshold=0.3):
    if not message or message.strip() == "":
        raise ValueError("Message cannot be empty!")
    
    message_vector = vectorizer.transform([message])
    prob = model.predict_proba(message_vector)[0][1]

    prediction = 1 if prob >= threshold else 0

    return prediction, prob

def main():
    """Main interactive loop for spam detection"""
    print("=" * 50)
    print("🔍 SMS SPAM DETECTOR")
    print("=" * 50)

    model,vectorizer = load_models()
    
    print("Type your message to check if it's spam.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            msg = input("📧 Enter Message: ").strip()

            if msg.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the spam detector!")
                break

            label, probability = predict_message(msg, model, vectorizer)

            print("\n" + "-" * 50)
            if label == 1:
                print(f"🚨 SPAM (Confidence: {probability:.2%})")
            else:
                print(f"✅ HAM (Confidence: {1-probability:.2%})")
            print("-" * 50 + "\n")

        except ValueError as e:
            print(f"⚠️  {e}\n")
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()