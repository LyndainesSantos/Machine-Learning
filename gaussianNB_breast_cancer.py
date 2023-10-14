from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = load_breast_cancer()

data = dataset.data
target = dataset.target

train, test, train_labels, test_labels = train_test_split(data, target, test_size=0.3, random_state=70)

modelo = GaussianNB()
modelo.fit(train, train_labels)

predicao = modelo.predict(test)

print("Acurácia: ", accuracy_score(test_labels, predicao))
print("Matriz de Confusão: ", confusion_matrix(test_labels, predicao))
print("Relatório: ", classification_report(test_labels, predicao))



