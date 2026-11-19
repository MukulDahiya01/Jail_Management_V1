"""Read-only synthetic demonstration API; not for legal, custody or bail decisions."""
import json
from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

ROOT=Path(__file__).resolve().parents[1]; MODEL_DIR=ROOT/'models'
app=FastAPI(title='Indian Jail Management | Synthetic AI/ML Demonstration',version='1.0.0',description='Fictional administrative review priority and occupancy forecasting. Not operational legal advice.')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

@lru_cache
def dashboard():
    return json.loads((MODEL_DIR/'dashboard.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=24)
def model(name):
    path=MODEL_DIR/name
    if not path.is_file(): raise HTTPException(status_code=503,detail='Model artifact not found. Run python train_models.py before launching.')
    return joblib.load(path)

class CaseReview(BaseModel):
    model_config=ConfigDict(extra='forbid')
    Number_of_Charges_Demo: int=Field(ge=0,le=100)
    Number_of_Accused_Demo: int=Field(ge=1,le=100)
    Prior_Convictions_Count_Demo: int=Field(ge=0,le=100)
    Pending_Days_Demo: int=Field(ge=0,le=10000)
    Hearings_Held_Demo: int=Field(ge=0,le=1000)
    Cognizable_Status: str
    Case_Stage_Demo: str
    Legal_Aid_Status_Demo: str
    Documentation_Status_Demo: str
    algorithm: str='Logistic Regression'

class Forecast(BaseModel):
    model_config=ConfigDict(extra='forbid')
    Authorized_Capacity: float=Field(gt=0,le=1000000)
    Population_Current: float=Field(ge=0,le=1000000)
    Population_Lag_1M: float=Field(ge=0,le=1000000)
    Population_Lag_3M: float=Field(ge=0,le=1000000)
    Population_Lag_6M: float=Field(ge=0,le=1000000)
    Admissions_Current_Month: float=Field(ge=0,le=100000)
    Releases_Current_Month: float=Field(ge=0,le=100000)
    Transfers_In_Current_Month: float=Field(ge=0,le=100000)
    Transfers_Out_Current_Month: float=Field(ge=0,le=100000)
    Calendar_Month: int=Field(ge=1,le=12)
    State_UT: str
    Facility_Type: str
    algorithm: str='Multiple Linear Regression'

CLASSIFIERS=['Logistic Regression','Decision Tree','KNN','SVM','Naive Bayes','Random Forest']
REGRESSORS=['Simple Linear Regression','Multiple Linear Regression','Decision Tree Regression','Support Vector Regression','Random Forest Regression']

def fname(name,suffix):return name.lower().replace(' ','_')+suffix

@app.get('/')
def index():return FileResponse(ROOT/'templates'/'index.html')

@app.get('/health')
def health():
    return {'status':'ok','models_ready':(MODEL_DIR/'dashboard.json').exists(),'synthetic_only':True}

@app.get('/api/dashboard')
def get_dashboard():return dashboard()

@app.get('/api/facilities')
def get_facilities(state: str|None=None,limit:int=Query(250,ge=1,le=1000)):
    rows=dashboard()['facilities']
    if state: rows=[r for r in rows if r['State_UT']==state]
    return {'facilities':rows[:limit],'count':len(rows)}

@app.post('/api/predict/review')
def review(data:CaseReview):
    if data.algorithm not in CLASSIFIERS:raise HTTPException(422,'Unknown classifier')
    payload=data.model_dump(); payload.pop('algorithm')
    for name in ['Cognizable_Status','Case_Stage_Demo','Legal_Aid_Status_Demo','Documentation_Status_Demo']:
        if payload[name] not in dashboard()['category_options'][name]:raise HTTPException(422,f'Unsupported {name} value')
    pipe=model(fname(data.algorithm,'_classifier.joblib'))
    frame=pd.DataFrame([payload]);label=str(pipe.predict(frame)[0]);result={'model':data.algorithm,'synthetic_case_review_priority':label,'warning':'Demonstration only: never use for bail, detention, sentencing, or allocation of real legal services.'}
    if hasattr(pipe,'predict_proba'):
      probs=pipe.predict_proba(frame)[0];result['class_probabilities']={str(k):round(float(v),4) for k,v in zip(pipe.classes_,probs)}
    return result

@app.post('/api/predict/forecast')
def forecast(data:Forecast):
    if data.algorithm not in REGRESSORS:raise HTTPException(422,'Unknown regression model')
    payload=data.model_dump();payload.pop('algorithm')
    for name in ['State_UT','Facility_Type']:
      if payload[name] not in dashboard()['reg_category_options'][name]:raise HTTPException(422,f'Unsupported {name} value')
    pipe=model(fname(data.algorithm,'.joblib'))
    forecast_value=max(0,float(pipe.predict(pd.DataFrame([payload]))[0]))
    return {'model':data.algorithm,'predicted_population_next_month':round(forecast_value,1),'predicted_occupancy_pct':round(100*forecast_value/data.Authorized_Capacity,1),'warning':'Synthetic demonstration forecast; not suitable for operational planning.'}

@app.get('/api/metadata')
def metadata():
    d=dashboard()
    return {'classifiers':CLASSIFIERS,'regressors':REGRESSORS,'category_options':d['category_options'],'reg_category_options':d['reg_category_options'],'synthetic_only':True}
