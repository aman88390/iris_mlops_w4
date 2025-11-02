from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("app/iris_model.pkl", "rb"))

@app.get("/")
def root():
    return {"message": "Iris API running on Kubernetes ✅"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open("app/iris_model.pkl", "rb"))

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris API running on Kubernetes ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: IrisFeatures):
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
