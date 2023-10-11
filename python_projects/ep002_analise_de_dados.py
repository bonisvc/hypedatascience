## IMPORTANDO OS PACOTES NECESSÁRIOS
from python_projects.binance_api import get_kline
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

## CARREGANDO OS DADOS
df_eth = get_kline(ticker="ETHUSDT")

## VERIFICANDO O SHAPE DA BASE DE DADOS CARREGADA E INFORMAÇÕES GERAIS
df_eth.info()
df_eth.head()

## SELECIONANDO APENAS AS COLUNAS NECESSÁRIAS PARA NOSSA ATIVIDADE
df_eth = df_eth[["timestamp", "close"]]

## CONVERTENDO A COLUNA DE TIMESTAMP PARA DATA E DO FECHAMENTO PARA NUMÉRICO
df_eth["timestamp"] = pd.to_datetime(df_eth["timestamp"], unit="ms")
df_eth["close"] = df_eth["close"].astype("float64")


### REALIZANDO VERIFICAÇÕES ESTATÍSTICAS COM TESTE DE BOX-JENKINS
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
plot_acf(df_eth["close"], ax=ax1, lags=40)
plot_pacf(df_eth["close"], ax=ax2, lags=40)
plt.show()

#VERIFICANDO TENDÊNCIA E SAZONALIDADE
decomposition = sm.tsa.seasonal_decompose(df_eth["close"]
                                          ,model="additive"
                                          ,period=365)

plt.figure(figsize=(15, 8))
plt.subplot(411)
plt.plot(np.array(df_eth["timestamp"]), np.array(df_eth["close"]), label="Original Prices")
plt.legend()
plt.subplot(412)
plt.plot(np.array(df_eth["timestamp"]), np.array(decomposition.trend), label="Trend")
plt.legend()
plt.subplot(413)
plt.plot(np.array(df_eth["timestamp"]), np.array(decomposition.seasonal), label="Seasonal")
plt.legend()
plt.subplot(414)
plt.plot(np.array(df_eth["timestamp"]), np.array(decomposition.resid), label="Residual")
plt.legend()
plt.tight_layout()
plt.show()

# TESTE DICKEY-FULLEY AUMENTADO PARA VERIFICAR ESTACIONARIDADE
def adf_test(df):
    results = sm.tsa.adfuller(df)
    
    statistic = results[0]
    p_value = results[1]
    critical_values = results[4]

    print(f"ADF Statistic: {statistic}")
    print(f"p-value: {p_value}")
    print("Critical values: ")
    for key, value in critical_values.items():
        print(f"{key}: {value}")

# hipótese nula (p-value > 0.05): a série não é estacionária
# hipótese alternativa (p-value < 0.05): a série é estacionária

adf_test(df_eth["close"])

#Primeira diferenciação (d=1)
diff1 = df_eth["close"] - df_eth["close"].shift(1)
diff1 = diff1.dropna()

adf_test(diff1) #1 diferenciação é o suficiente

params=(1, 1, 1) ## (p, d, q)
model = ARIMA(df_eth['close'], order=params)
results = model.fit()
residuals = results.resid

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(residuals, ax=ax1, lags=40)
plot_pacf(residuals, ax=ax2, lags=40)
plt.show()

previsao = results.forecast(7)

max_timestamp = df_eth["timestamp"].max() + pd.DateOffset(days=1)
date_range = pd.date_range(start=max_timestamp ,periods=7)

df_previsao = pd.DataFrame({"timestamp": date_range, "close": previsao})

df_last_7_day = df_eth[-7:]

plt.plot(np.array(df_last_7_day["timestamp"]), np.array(df_last_7_day["close"]), label="real")
plt.legend()
plt.plot(np.array(df_previsao["timestamp"]), np.array(df_previsao["close"]), label="predicted")
plt.legend()
plt.show()
