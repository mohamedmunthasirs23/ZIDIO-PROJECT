import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_price_trends(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index, y=df['price'])
    plt.title("Cryptocurrency Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('crypto_data.csv', index_col=0, parse_dates=True)
    plot_price_trends(df)
