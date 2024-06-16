from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

FILENAME = "titanic_voting_classifier.pkl"

app = FastAPI()


class model_props(BaseModel):
    sex: int
    age: int
    family_size: int
    pclass: int


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isfile(f"Model/{FILENAME}"):
    print("Loading the model")
    best_xgb = joblib.load(f"Model/{FILENAME}")
else:
    print("Model doesn't exist")
    exit()


@app.get("/health")
async def health_check():
    return {"message": "API is working!"}


@app.get("/api/predict")
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
                "family_size": "FamilySize",
                "pclass": "Pclass",
            }
        )
        return {"data": str(best_xgb.predict(df)[0])}
    else:
        raise HTTPException(status_code=500, detail="Model_props are not correct")
