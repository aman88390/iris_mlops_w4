from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

pickle.dump(model, open("app/iris_model.pkl", "wb"))
print("✅ Model saved as app/iris_model.pkl")
