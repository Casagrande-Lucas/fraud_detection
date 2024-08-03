import pandas as pd


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df):
    df = df.dropna()
    df['normalized_amount'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    return df


if __name__ == "__main__":
    df = load_data('../data/transactions.csv')
    df = preprocess_data(df)
    df.to_csv('../data/transactions_preprocessed.csv', index=False)
