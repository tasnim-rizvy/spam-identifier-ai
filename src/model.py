from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import numpy as np

class SpamClassifier:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)
        self.is_trained = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test, threshold=0.5):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        probs = self.model.predict_proba(X_test)
        spam_probs = probs[:, 1]

        return (spam_probs >= threshold).astype(int)
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        predictions = self.predict(X_test, threshold)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        cm = confusion_matrix(y_test, predictions)

        report = classification_report(y_test, predictions)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }
    
    def get_feature_importance(self, vectorizer, top_n=20):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        feature_names = vectorizer.get_feature_names_out()

        log_probs = self.model.feature_log_prob_

        spam_indices = np.argsort(log_probs[1])[-top_n:][::-1]
        spam_words = [(feature_names[i], log_probs[1][i]) for i in spam_indices]

        ham_indices = np.argsort(log_probs[0])[-top_n:][::-1]
        ham_words = [(feature_names[i], log_probs[0][i]) for i in ham_indices]
        
        return spam_words, ham_words