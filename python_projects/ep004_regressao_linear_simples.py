## IMPORTANDO OS PACOTES NECESSÁRIOS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

dir = os.getcwd()

## CARREGANDO OS DADOS
df_salary = pd.read_csv(f"{dir}/python_projects/database/salary.csv"
                        ,sep=","
                        ,index_col=0)

## VERIFICANDO O SHAPE DA BASE DE DADOS CARREGADA E INFORMAÇÕES GERAIS
df_salary.info()
df_salary.head()
df_salary.describe()

## ALTERANDO OS NOMES DAS COLUNAS
df_salary = df_salary.rename(columns={"YearsExperience": "experience"
                                      ,"Salary": "salary"})

### PLOTANDO UM SCATTER PARA ANALISAR VISUALMENTE A RELAÇÃO LINEAR ENTRE AS VARIÁVEIS
plt.figure(figsize=[15, 8])
plt.scatter(df_salary["experience"].values
         ,df_salary["salary"].values)
plt.xlabel("anos de experiência")
plt.ylabel("salário")
plt.title("Salário por Anos de Experiência")
plt.show()

### VERIFICANDO SE EXISTE REALMENTE CORRELAÇÃO ENTRE AS VARIÁVEIS
correlation = df_salary["salary"].corr(df_salary["experience"])
print(f"Coeficiente de correlação: {correlation:.2f}")

# Calculando o t de Student
graus_de_liberdade = len(df_salary) - 2
t_score = (correlation * 
           np.sqrt(graus_de_liberdade) / np.sqrt(1 - correlation ** 2))

# Calculando o p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_score), df=graus_de_liberdade))

# Analisando o p_value
print(f"p-value: {p_value}")

# Normalizando a amostra
scaler = StandardScaler()
df_salary["salary"] = scaler.fit_transform(df_salary["salary"].values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(
    df_salary["experience"]
    ,df_salary["salary"]
    ,test_size=0.25
    ,random_state=423
)

# Criando o modelo de regressão
model = LinearRegression()
# Ajustando o modelo às amostras de treino
model.fit(x_train.values.reshape(-1, 1), y_train.values)

# Predizendo valores de Y com base na amostra de teste de X
y_pred = model.predict(x_test.values.reshape(-1, 1))

# Verificando o erro e o r²
mse = mean_squared_error(y_test.values, y_pred)
r2 = r2_score(y_test.values, y_pred)

print(f"MSE: {mse:.2f} \n r²: {r2:.2f}")

y_plot = model.predict(df_salary["experience"].values.reshape(-1, 1))

plt.figure(figsize=[15, 8])
plt.scatter(df_salary["experience"].values, df_salary["salary"].values, label="observações")
plt.plot(df_salary["experience"].values, y_plot, label="previsto")
plt.show()

x_predict = np.array([2, 5, 6]).reshape(-1, 1)
y_predict = model.predict(x_predict)

plt.figure(figsize=[15, 8])
plt.scatter(df_salary["experience"].values, df_salary["salary"].values, label="observações")
plt.plot(df_salary["experience"].values, y_plot, label="previsto")
plt.scatter(x_predict, y_predict, label="novas_previsões", color="r")
plt.show()