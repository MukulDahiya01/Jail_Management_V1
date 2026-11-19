from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)

def test_health():
    response=client.get('/health')
    assert response.status_code==200
    assert response.json()['models_ready'] is True

def test_dashboard():
    r=client.get('/api/dashboard')
    assert r.status_code==200
    assert len(r.json()['classification'])==6
    assert len(r.json()['regression'])==5

def test_review_prediction():
    data={'Number_of_Charges_Demo':2,'Number_of_Accused_Demo':2,'Prior_Convictions_Count_Demo':1,'Pending_Days_Demo':120,'Hearings_Held_Demo':3,'Cognizable_Status':'Cognizable','Case_Stage_Demo':'Trial','Legal_Aid_Status_Demo':'Assigned','Documentation_Status_Demo':'Complete','algorithm':'Logistic Regression'}
    meta=client.get('/api/metadata').json()['category_options']
    for k in ['Cognizable_Status','Case_Stage_Demo','Legal_Aid_Status_Demo','Documentation_Status_Demo']:
        data[k]=meta[k][0]
    r=client.post('/api/predict/review',json=data)
    assert r.status_code==200, r.text
    assert r.json()['synthetic_case_review_priority'] in ('High','Standard')

def test_forecast():
    meta=client.get('/api/metadata').json()['reg_category_options']
    payload={'Authorized_Capacity':500,'Population_Current':450,'Population_Lag_1M':440,'Population_Lag_3M':430,'Population_Lag_6M':420,'Admissions_Current_Month':35,'Releases_Current_Month':30,'Transfers_In_Current_Month':6,'Transfers_Out_Current_Month':5,'Calendar_Month':6,'State_UT':meta['State_UT'][0],'Facility_Type':meta['Facility_Type'][0]}
    r=client.post('/api/predict/forecast',json=payload)
    assert r.status_code==200,r.text
    assert r.json()['predicted_population_next_month']>=0
