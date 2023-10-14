from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

dataset = load_diabetes()

data = dataset.data
target = dataset.target

train, test, train_labels, test_labels = train_test_split(data, target, test_size=0.2, random_state=110)

modelo = LinearRegression()
modelo.fit(train, train_labels)

print("Coeficientes: ", modelo.coef_)
print("Variância (R^2): ", modelo.score(test, test_labels))
predicao = modelo.predict(test)
erro_residual = test_labels - predicao

plt.scatter(range(len(erro_residual)), erro_residual, color='blue', label='E Res.')
plt.axhline(y = 0, color='red', linewidth=2)
plt.title("Erro Residual", size = 20)
plt.xlabel('Índice')
plt.ylabel('Erro Residual')
plt.legend()
plt.show()