import pandas as pd

def load_data(path: str):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df