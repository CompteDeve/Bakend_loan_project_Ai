from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
import json
from typing import Dict, List
import datetime

app = FastAPI(title="FinScore AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle
try:
    model = joblib.load('loan_approval_model.joblib')
    print("✅ Modèle chargé avec succès")
    print(f"📊 Features attendues: {model.feature_names_in_}")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle: {e}")
    model = None

class LoanData(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: float
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    credit_score: int
    cb_person_cred_hist_length: float
    previous_loan_defaults_on_file: str

class AnalysisResult(BaseModel):
    status: int
    probability: float
    confidence: float
    risk_factors: Dict[str, float]
    recommendation: str
    timestamp: str

@app.get("/")
async def read_root():
    return {
        "service": "FinScore AI - Credit Analysis API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/predict - POST",
            "health": "/health - GET",
            "stats": "/stats - GET"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/stats")
async def get_statistics():
    """Retourne des statistiques pour les graphiques"""
    return {
        "education_distribution": {
            "High School": 15,
            "Associate": 25,
            "Bachelor": 35,
            "Master": 20,
            "Doctorate": 5
        },
        "approval_rates": {
            "PERSONAL": 62,
            "EDUCATION": 78,
            "MEDICAL": 55,
            "VENTURE": 45,
            "DEBTCONSOLIDATION": 65,
            "HOMEIMPROVEMENT": 72
        },
        "risk_factors": [
            "credit_score",
            "loan_to_income_ratio",
            "employment_experience",
            "age",
            "interest_rate",
            "credit_history"
        ]
    }

@app.post("/predict", response_model=AnalysisResult)
async def predict(data: LoanData):
    if model is None:
        return AnalysisResult(
            status=0,
            probability=0.0,
            confidence=0.0,
            risk_factors={},
            recommendation="Modèle non disponible",
            timestamp=datetime.datetime.now().isoformat()
        )
    
    try:
        # 1. Convertir en DataFrame
        df = pd.DataFrame([data.dict()])
        
        # 2. Calculer les features supplémentaires
        df['loan_percent_income'] = df['loan_amnt'] / df['person_income'] if data.person_income > 0 else 0.0
        
        # 3. Encodage
        gender_map = {'male': 1, 'female': 0}
        education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
        home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
        intent_map = {
            'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2,
            'VENTURE': 3, 'DEBTCONSOLIDATION': 4, 'HOMEIMPROVEMENT': 5
        }
        default_map = {'No': 0, 'Yes': 1}
        
        df['person_gender'] = df['person_gender'].map(gender_map).fillna(0)
        df['person_education'] = df['person_education'].map(education_map).fillna(0)
        df['person_home_ownership'] = df['person_home_ownership'].map(home_map).fillna(0)
        df['loan_intent'] = df['loan_intent'].map(intent_map).fillna(0)
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(default_map).fillna(0)
        
        # 4. Features attendues
        expected_features = [
            'person_age', 'person_gender', 'person_education', 'person_income',
            'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file'
        ]
        
        # 5. Préparation
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        df = df[expected_features]
        
        # 6. Prédiction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        probability = float(probabilities[1])
        
        # 7. Calcul des facteurs de risque
        risk_factors = {
            "credit_score": min(100, (data.credit_score / 850) * 100),
            "loan_to_income_ratio": min(100, ((data.loan_amnt / data.person_income) * 100) * 3),
            "employment_experience": min(100, (data.person_emp_exp / 30) * 100),
            "age_risk": 100 - abs(data.person_age - 35) if 25 <= data.person_age <= 55 else 50,
            "interest_rate": min(100, (data.loan_int_rate / 25) * 100),
            "credit_history": min(100, (data.cb_person_cred_hist_length / 20) * 100)
        }
        
        # 8. Recommandation
        if probability > 0.7:
            recommendation = "Demande fortement recommandée"
        elif probability > 0.5:
            recommendation = "Demande recommandée avec conditions"
        elif probability > 0.3:
            recommendation = "Demande à examiner manuellement"
        else:
            recommendation = "Demande non recommandée"
        
        return AnalysisResult(
            status=int(prediction),
            probability=probability,
            confidence=float(probabilities.max()),
            risk_factors=risk_factors,
            recommendation=recommendation,
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        print(f"❌ ERREUR: {str(e)}\n{traceback.format_exc()}")
        
        return AnalysisResult(
            status=0,
            probability=0.0,
            confidence=0.0,
            risk_factors={},
            recommendation=f"Erreur: {str(e)}",
            timestamp=datetime.datetime.now().isoformat()
        )

@app.post("/batch_predict")
async def batch_predict(data: List[LoanData]):
    """Prédiction par lot pour l'historique"""
    results = []
    for item in data:
        result = await predict(item)
        results.append(result.dict())
    return {"results": results}