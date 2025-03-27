import requests
import pandas as pd

def fetch_crypto_data(crypto='bitcoin', days=365):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days={days}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.to_csv('crypto_data.csv')  # Save data for further use
    return df

if __name__ == "__main__":
    df = fetch_crypto_data()
    print(df.head())
