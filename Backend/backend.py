from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

FILENAME = "titanic_voting_classifier.pkl"
MODEL_PATH = f"/Model/{FILENAME}"

if os.path.isfile(MODEL_PATH):
    print("Loading the model")
    best_xgb = joblib.load(MODEL_PATH)
else:
    print("Model doesn't exist")
    exit()

app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")


class model_props(BaseModel):
    sex: int
    age: int
    familySize: int
    pclass: int


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"message": "API is working!"}


@app.post("/api/predict")
async def get_prediction(model_props: model_props):
    if model_props:
        model_props_dict = jsonable_encoder(model_props)
        df = pd.DataFrame(
            data=[model_props_dict.values()], columns=model_props_dict.keys()
        )
        df = df.rename(
            columns={
                "sex": "Sex",
                "age": "Age",
                "familySize": "FamilySize",
                "pclass": "Pclass",
            }
        )
        return {"data": str(best_xgb.predict(df)[0])}
    else:
        raise HTTPException(status_code=500, detail="Model props are not correct")
