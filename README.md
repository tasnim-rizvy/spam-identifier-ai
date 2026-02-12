# 📧 SMS Spam Classifier

A beginner-friendly spam detection system using Machine Learning!

## 🎯 What Does This Project Do?

This project classifies text messages as either:
- **SPAM** 🚨 - Unwanted messages (ads, scams, promotions)
- **HAM** ✅ - Normal, legitimate messages

It uses **Naive Bayes**, a simple but powerful machine learning algorithm that works great for text classification!

## 📁 Project Structure

```
spam-classifier/
├── data/
│   └── SMSSpamCollection          # Your dataset (tab-separated file)
├── models/
│   ├── spam_model.pkl             # Trained model (created after training)
│   └── vectorizer.pkl             # TF-IDF vectorizer (created after training)
├── src/
│   ├── data/
│   │   ├── loader.py              # Loads the dataset
│   │   └── preprocess.py          # Cleans the text
│   ├── features/
│   │   └── feature_extraction.py  # Converts text to numbers (TF-IDF)
│   └── model.py                   # The classifier model
├── train.py                       # Script to train the model
├── predict.py                     # Script to test messages
├── test_classifier.py             # Comprehensive testing suite
└── requirements.txt               # Python dependencies
```

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Make sure you have the `SMSSpamCollection` file in the `data/` folder.

The file should look like this:
```
ham    Hey, how are you?
spam   WINNER!! You've won a prize! Call now!
ham    Can you pick up milk?
```

### Step 3: Train the Model

```bash
python train.py
```

This will:
- Load your data
- Clean and process the text
- Train the spam classifier
- Save the trained model
- Show you performance metrics

**Expected Output:**
```
✅ Loaded 5574 messages
   - Spam messages: 747 (13.4%)
   - Ham messages: 4827 (86.6%)
✅ Models saved successfully!
📈 Accuracy: 98.5%
```

### Step 4: Test the Model

Run the comprehensive test suite:
```bash
python test_classifier.py
```

This will test:
- Basic spam/ham detection
- Edge cases (empty messages, numbers, etc.)
- Different threshold sensitivity
- Feature importance (which words indicate spam)
- Custom message testing

### Step 5: Use the Predictor

```bash
python predict.py
```

Then type messages to check if they're spam:
```
📧 Enter Message: Congratulations! You won a prize!
🚨 SPAM (Confidence: 95.67%)

📧 Enter Message: Hey, are we still meeting for lunch?
✅ HAM (Confidence: 97.23%)
```

## 🔧 How It Works (Simple Explanation)

### 1. **Text Preprocessing**
- Converts text to lowercase: "HELLO" → "hello"
- Removes numbers: "call 123" → "call"
- Removes punctuation: "Hello!" → "Hello"
- Removes extra spaces

### 2. **Feature Extraction (TF-IDF)**
- Converts words to numbers that computers can understand
- TF-IDF = "Term Frequency - Inverse Document Frequency"
- Important words get higher scores
- Common words (like "the", "a") get lower scores

### 3. **Naive Bayes Classification**
- Learns patterns from training data
- Calculates probability: "How likely is this spam?"
- Makes predictions based on word patterns

### 4. **Threshold**
- Default: 0.3 (30% confidence)
- Lower threshold = More sensitive (catches more spam, but more false alarms)
- Higher threshold = Less sensitive (misses some spam, but fewer false alarms)

## 📊 Understanding the Metrics

When you train the model, you'll see these metrics:

- **Accuracy**: How often the model is correct overall (aim for 95%+)
- **Precision**: Of all messages marked as spam, how many are actually spam? (aim for 95%+)
- **Recall**: Of all actual spam messages, how many did we catch? (aim for 90%+)
- **F1 Score**: Balance between precision and recall (aim for 92%+)

### Confusion Matrix:
```
              Predicted
              Ham    Spam
Actual Ham    950    10     ← 10 false positives (ham marked as spam)
       Spam   5      150    ← 5 false negatives (spam missed)
```

## 🎓 Key Concepts for Beginners

### What is Machine Learning?
Teaching computers to learn from examples instead of programming every rule.

### What is Training?
Showing the computer many examples so it can learn patterns:
- "FREE" often appears in spam
- "meeting" often appears in normal messages

### What is a Model?
The "brain" of your program - it stores what was learned during training.

### What is TF-IDF?
A way to represent text as numbers. Words that are:
- Common everywhere → Low score
- Unique and informative → High score

## 🐛 Common Issues & Solutions

### Issue: "Model file not found"
**Solution:** Run `python train.py` first to create the model!

### Issue: "Data file not found"
**Solution:** Make sure `data/SMSSpamCollection` exists in the correct location.

### Issue: Poor accuracy (below 90%)
**Solution:** 
- Check if your data file is formatted correctly
- Make sure you have enough training data
- Try adjusting the `alpha` parameter in `train.py`

### Issue: Too many false positives
**Solution:** Increase the threshold in `predict.py` (e.g., from 0.3 to 0.5)

## 🎯 Improvements You Made

### Original Issues Fixed:
1. ✅ Fixed bug: `msg.lower` → `msg.lower()`
2. ✅ Added error handling for missing files
3. ✅ Added input validation
4. ✅ Better user interface with emojis
5. ✅ Comprehensive testing suite
6. ✅ Better documentation

### New Features Added:
- Interactive testing mode
- Detailed evaluation metrics
- Feature importance analysis
- Edge case testing
- Beginner-friendly explanations

## 🚀 Next Steps to Learn More

1. **Experiment with thresholds**: Try different values (0.2, 0.3, 0.5) and see how it affects results
2. **Test with your own messages**: Use the interactive mode to test real messages
3. **Understand the math**: Research how Naive Bayes works
4. **Try other algorithms**: Experiment with Logistic Regression or Random Forest
5. **Add more features**: Extract message length, number of capitals, exclamation marks

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [What is Naive Bayes?](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## 🤝 Need Help?

If you get stuck:
1. Read the error message carefully
2. Check if all files are in the right place
3. Make sure you ran `train.py` before `predict.py`
4. Try running `test_classifier.py` to diagnose issues

## 🎉 Congratulations!

You've built a real machine learning project! This is the same technology used by:
- Email services (Gmail, Outlook) to filter spam
- Social media platforms to detect fake accounts
- Banks to detect fraud

Keep learning and experimenting! 🚀