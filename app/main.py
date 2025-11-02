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
