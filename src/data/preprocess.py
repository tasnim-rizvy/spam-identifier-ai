from src.data.loader import load_data
import re, os

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def processed_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "../../data/SMSSpamCollection")

    data_frame = load_data(DATA_PATH)
    data_frame['clean_message'] = data_frame['message'].apply(clean_text)

    return data_frame