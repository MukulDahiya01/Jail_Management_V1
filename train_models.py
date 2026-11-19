"""Reproducible offline training. All data and predictions are synthetic demos."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.dummy import DummyClassifier

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'Indian_Jail_Management_Synthetic_Expanded_Validated.xlsx'
MODELS=ROOT/'models'; MODELS.mkdir(exist_ok=True)
NUM=['Number_of_Charges_Demo','Number_of_Accused_Demo','Prior_Convictions_Count_Demo','Pending_Days_Demo','Hearings_Held_Demo']
CAT=['Cognizable_Status','Case_Stage_Demo','Legal_Aid_Status_Demo','Documentation_Status_Demo']
RNUM=['Authorized_Capacity','Population_Current','Population_Lag_1M','Population_Lag_3M','Population_Lag_6M','Admissions_Current_Month','Releases_Current_Month','Transfers_In_Current_Month','Transfers_Out_Current_Month','Calendar_Month']
RCAT=['State_UT','Facility_Type']

def prep(nums,cats,nonnegative=False):
    scaler=MinMaxScaler() if nonnegative else StandardScaler()
    return ColumnTransformer([
        ('numbers',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',scaler)]),nums),
        ('categories',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)
    ])

def excel_date(s):
    if pd.api.types.is_numeric_dtype(s): return pd.to_datetime(s,unit='D',origin='1899-12-30')
    return pd.to_datetime(s)

def main():
    sheets=pd.read_excel(DATA,sheet_name=['Facility_Master_Demo','Prison_Monthly_Demo','Regression_Training_Demo','Classification_Cases_Demo'])
    fac=sheets['Facility_Master_Demo']; monthly=sheets['Prison_Monthly_Demo'].copy(); reg=sheets['Regression_Training_Demo'].copy(); cls=sheets['Classification_Cases_Demo'].copy()
    monthly['Month']=excel_date(monthly['Month']); reg['Feature_Month']=excel_date(reg['Feature_Month'])
    monthly['Month']=monthly['Month'].dt.strftime('%Y-%m-%d')
    x=cls[NUM+CAT].copy(); y=cls['Case_Review_Priority_Synthetic'].astype(str)
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=.25,random_state=42,stratify=y)
    classifiers={
        'Logistic Regression':LogisticRegression(max_iter=1500,class_weight='balanced'),
        'Decision Tree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,random_state=42,class_weight='balanced'),
        'KNN':KNeighborsClassifier(n_neighbors=11),
        'SVM':SVC(kernel='rbf',C=1,class_weight='balanced',probability=True),
        'Naive Bayes':MultinomialNB(),
        'Random Forest':RandomForestClassifier(n_estimators=150,max_depth=8,min_samples_leaf=5,class_weight='balanced',random_state=42),
    }
    scores=[]
    for name,est in classifiers.items():
        m=Pipeline([('preprocess',prep(NUM,CAT,name=='Naive Bayes')),('model',est)])
        m.fit(xtr,ytr); p=m.predict(xte)
        scores.append({'name':name,'accuracy':round(float(accuracy_score(yte,p)),4),'f1':round(float(f1_score(yte,p,average='weighted',zero_division=0)),4)})
        joblib.dump(m,MODELS/(name.lower().replace(' ','_')+'_classifier.joblib'),compress=3)
    dummy=DummyClassifier(strategy='most_frequent').fit(xtr,ytr)
    baseline=float(accuracy_score(yte,dummy.predict(xte)))
    reg=reg.sort_values('Feature_Month').dropna(subset=['Target_Population_Next_Month']).copy()
    months=sorted(reg['Feature_Month'].unique()); split=months[int(len(months)*.8)]
    tr=reg[reg.Feature_Month<split]; te=reg[reg.Feature_Month>=split]
    Xtr=tr[RNUM+RCAT]; Xte=te[RNUM+RCAT]
    Ytr=tr.Target_Population_Next_Month; Yte=te.Target_Population_Next_Month
    regressors={
        'Simple Linear Regression':(LinearRegression(),['Population_Current'],[]),
        'Multiple Linear Regression':(LinearRegression(),RNUM,RCAT),
        'Decision Tree Regression':(DecisionTreeRegressor(max_depth=8,min_samples_leaf=8,random_state=42),RNUM,RCAT),
        'Support Vector Regression':(SVR(kernel='rbf',C=10,epsilon=5),RNUM,RCAT),
        'Random Forest Regression':(RandomForestRegressor(n_estimators=150,max_depth=10,min_samples_leaf=5,random_state=42,n_jobs=-1),RNUM,RCAT)
    }
    reg_scores=[]
    for name,(est,n,c) in regressors.items():
        model=Pipeline([('preprocess',prep(n,c)),('model',est)]).fit(Xtr[n+c],Ytr)
        pr=model.predict(Xte[n+c]); reg_scores.append({'name':name,'mae':round(float(mean_absolute_error(Yte,pr)),2),'r2':round(float(r2_score(Yte,pr)),4)})
        joblib.dump(model,MODELS/(name.lower().replace(' ','_')+'.joblib'),compress=3)
    # Facility-level clustering (same derived feature family as notebook)
    group=monthly.groupby('Facility_ID').agg(Mean_Occupancy=('Occupancy_Rate_pct_Demo','mean'),Mean_Population=('Inmate_Population_Demo','mean'),Mean_Admissions=('Admissions_Demo','mean'),Mean_Releases=('Releases_Demo','mean')).reset_index()
    feature_names=['Mean_Occupancy','Mean_Population','Mean_Admissions','Mean_Releases']; scaler=StandardScaler().fit(group[feature_names]); scaled=scaler.transform(group[feature_names]); km=KMeans(n_clusters=3,random_state=42,n_init=10).fit(scaled)
    group['cluster']=km.labels_.astype(int); joblib.dump({'scaler':scaler,'model':km,'features':feature_names},MODELS/'facility_clustering.joblib')
    # Save compact dashboards data rather than loading Excel per request.
    merged=group.merge(fac[['Facility_ID','Facility_Name_Demo','State_UT','Facility_Type_Demo','Authorized_Capacity_Demo']],on='Facility_ID',how='left')
    trend=monthly.groupby('Month',as_index=False).agg(population=('Inmate_Population_Demo','sum'),capacity=('Authorized_Capacity_Demo','sum'),admissions=('Admissions_Demo','sum'),releases=('Releases_Demo','sum'))
    trend['occupancy_pct']=(100*trend.population/trend.capacity).round(2)
    states=monthly.groupby('State_UT',as_index=False).agg(mean_occupancy=('Occupancy_Rate_pct_Demo','mean'),facilities=('Facility_ID','nunique'));states['mean_occupancy']=states.mean_occupancy.round(2)
    latest=monthly.sort_values('Month').groupby('Facility_ID').tail(1)
    overview={'facilities':int(len(fac)),'cases':int(len(cls)),'monthly_records':int(len(monthly)),'regression_records':int(len(reg)),'latest_month':str(monthly.Month.max()),'latest_population':int(latest.Inmate_Population_Demo.sum()),'latest_capacity':int(latest.Authorized_Capacity_Demo.sum()),'average_occupancy_pct':round(float(latest.Inmate_Population_Demo.sum()/latest.Authorized_Capacity_Demo.sum()*100),2)}
    # Association rules on facility-level indicators: meaningful within-facility co-occurrence, synthetic only.
    frame=group.copy(); frame['High_Occupancy']=frame.Mean_Occupancy>=frame.Mean_Occupancy.median(); frame['High_Admissions']=frame.Mean_Admissions>=frame.Mean_Admissions.median(); frame['High_Population']=frame.Mean_Population>=frame.Mean_Population.median(); frame['High_Releases']=frame.Mean_Releases>=frame.Mean_Releases.median()
    flags=['High_Occupancy','High_Admissions','High_Population','High_Releases']; rules=[]
    for left in flags:
      for right in flags:
        if left==right: continue
        both=(frame[left]&frame[right]).mean(); support_a=frame[left].mean(); support_b=frame[right].mean()
        if both>=.15 and support_a>0 and support_b>0:
          rules.append({'antecedent':left.replace('_',' '),'consequent':right.replace('_',' '),'support':round(float(both),3),'confidence':round(float(both/support_a),3),'lift':round(float(both/support_a/support_b),3)})
    rules=sorted(rules,key=lambda r:r['lift'],reverse=True)
    payload={'overview':overview,'classification':scores,'classification_baseline':round(baseline,4),'regression':reg_scores,'facilities':merged.round(3).to_dict('records'),'trends':trend.to_dict('records'),'states':states.to_dict('records'),'associations':rules[:20], 'category_options':{c:sorted([str(v) for v in cls[c].dropna().unique()]) for c in CAT},'reg_category_options':{c:sorted([str(v) for v in reg[c].dropna().unique()]) for c in RCAT},'data_note':'Entirely simulated prison and case data; no real legal or detention decision-making.'}
    (MODELS/'dashboard.json').write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
    print('Training complete:',overview,'classifier scores:',scores)

if __name__=='__main__':main()
