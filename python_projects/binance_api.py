import pandas as pd
import requests


def get_kline(ticker="BTCUSDT", timeframe="1d"):
    '''
    code to connect within binance api and return the kline of specified market

    :param ticker: market pair symbol of coin or exchange. ex.: 'BTCUSDT'
    :param timeframe: slice of time to return the kline. ex.: '1d = daily'
    :return: dataframe with the last 1000 kline interval information
    '''

    endpoint = "https://api.binance.com/api/v1/klines"
    params = {
        "symbol": ticker,
        "interval": timeframe,
        "limit": 1000,
    }
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        # convert json file to pandas dataframe
        data = response.json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                         "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
                                         "taker_buy_quote_asset_volume", "ignore"])

        return df
    else:
        print(f"Erro na requisição: {response.status_code} - {response.text}")