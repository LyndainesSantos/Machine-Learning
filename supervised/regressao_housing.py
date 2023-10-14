from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import joblib

# Carrega os dados do dataset fetch_california_housing
dados_entrada = datasets.fetch_california_housing()

data = dados_entrada.data # Conjunto de dados de entrada (características/atributos)
target = dados_entrada.target # Rótulo de cada dado do conjunto de entrada em X

# Separação do conjunto de dados para treino (train) e para teste (test) e os respectivos rótulos (train_labels, y_test_labels)
train, test, train_labels, test_labels = train_test_split(data, target, train_size=0.3, random_state=1)

modelo  = LinearRegression()
modelo.fit(train, train_labels)

# Salvar Modelo
joblib.dump(modelo, 'modelo_regressao.pkl')
print("Modelo Salvo!")

# Visualizar os coeficientese 
print("Coeficientes:", modelo.coef_)
print("Variância (R^2):", modelo.score(test, test_labels))

# Predição dos modelos a partir dos dados de teste
predicao = modelo.predict(test)

# Calculo do erro residual dos dados
erro_residual = test_labels - predicao
print(erro_residual)

# Visualização do erro residual
plt.scatter(range(len(erro_residual)), erro_residual)
plt.title("Erro Residual", size=24)
plt.axhline(y = 0, color='red')
plt.xlabel("Índice")
plt.ylabel("Erro Residual")
plt.show()