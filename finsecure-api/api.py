import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 1. Inicializar la App
app = FastAPI(
    title="FinSecure Fraud Detection API",
    description="API de inferencia para detección de fraude bancario usando MLP.",
    version="1.0.0"
)

# 2. Definir el Esquema de Entrada (Contrato de Datos)
class TransactionInput(BaseModel):
    transaction_id: str
    # Esperamos una lista de 29 valores: V1 a V28 + Amount
    features: List[float] 

# 3. Cargar Artefactos al Inicio
# Usamos variables globales para cargar el modelo una sola vez en memoria
model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        # Asegúrate de que los nombres de archivo coincidan con los tuyos
        model = joblib.load("modelo_mlp_fraude.joblib")
        scaler = joblib.load("scaler_fraude.joblib")
        print("✅ Artefactos cargados exitosamente.")
    except Exception as e:
        print(f"❌ Error al cargar artefactos: {e}")

# 4. Endpoint de Predicción
@app.post("/predict", tags=["Inferencia"])
def predict_fraud(transaction: TransactionInput):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")

    # Validar dimensiones (28 componentes PCA + 1 Amount = 29)
    if len(transaction.features) != 29:
        raise HTTPException(
            status_code=400, 
            detail=f"Se esperaban 29 características, recibidas: {len(transaction.features)}"
        )

    try:
        # --- LÓGICA CRÍTICA DE TU PROYECTO ---
        # Reconstruir DataFrame con nombres exactos para evitar el warning del StandardScaler
        # Referencia: Tu archivo mockup_finsecure.py
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Crear DataFrame de una sola fila
        input_df = pd.DataFrame([transaction.features], columns=feature_names)
        
        # 1. Escalar datos
        input_scaled = scaler.transform(input_df)
        
        # 2. Predecir probabilidad (clase 1 = Fraude)
        # predict_proba devuelve [[prob_no_fraude, prob_fraude]]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # 3. Aplicar Umbral de Negocio (0.20 según tu informe)
        THRESHOLD = 0.20
        prediction_label = "FRAUDE" if probability >= THRESHOLD else "NORMAL"
        
        return {
            "transaction_id": transaction.transaction_id,
            "prediction": prediction_label,
            "risk_score": round(float(probability), 4),
            "threshold_used": THRESHOLD,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en inferencia: {str(e)}")

# Endpoint de prueba básico
@app.get("/")
def home():
    return {"message": "FinSecure API is running"}