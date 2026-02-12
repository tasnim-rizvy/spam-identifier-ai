from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.preprocess import processed_data

def extract_features():
    data_frame = processed_data()

    X_message = data_frame['clean_message']
    y = data_frame['label']

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english",
        max_df=0.95,
        min_df=2
    )

    X = vectorizer.fit_transform(X_message)

    return X, y, vectorizer