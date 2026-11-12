from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

# Загружаем данные
iris = load_iris()
X, y = iris.data, iris.target

# Обучаем модель (дерево решений)
model = DecisionTreeClassifier()
model.fit(X, y)

# Сохраняем модель в файл
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель обучена и сохранена в model.pkl")