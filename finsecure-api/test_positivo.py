import requests
import json

# URL de tu API local
url = "http://127.0.0.1:8000/predict"

# Datos reales de una transacción NORMAL (Clase 0 - Row 0 del dataset)
# V1...V28 + Amount (149.62)
payload_normal = {
    "transaction_id": "TRX-NORMAL-001",
    "features": [
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 
        0.462388, 0.239599, 0.098698, 0.363787, 0.090794, 
        -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, 
        -0.470401, 0.207971, 0.025791, 0.403993, 0.251412, 
        -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, 
        -0.189115, 0.133558, -0.021053, 149.62
    ]
}

try:
    print(f"📡 Enviando transacción legítima a {url}...")
    response = requests.post(url, json=payload_normal)
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ RESPUESTA DE LA API:")
        print(json.dumps(data, indent=2))
        
        # Validación visual rápida
        if data["prediction"] == "NORMAL":
            print("\n🎉 ¡CORRECTO! El modelo identificó que NO es fraude.")
        else:
            print("\n⚠️ OJO: El modelo la marcó como fraude (Falso Positivo).")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"❌ Error de conexión: {e}")