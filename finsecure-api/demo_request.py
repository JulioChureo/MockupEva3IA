import requests
import json

# Esta es la dirección donde "vive" tu API localmente
url = "http://127.0.0.1:8000/predict"

# Datos de ejemplo de una transacción fraudulenta real del dataset
payload = {
    "transaction_id": "TRX-PRUEBA-001",
    "features": [
        -2.31, 1.98, -1.60, 3.99, -0.52, -1.42, -2.53, 1.39, -2.77, -2.77, 
        3.20, -2.89, -0.59, -4.28, 0.38, -1.14, -2.83, -0.01, 0.41, 0.12, 
        0.51, -0.03, -0.46, 0.32, 0.04, 0.17, 0.26, -0.14, 0.00
    ]
}

try:
    # Enviamos la petición POST a la API
    print(f"📡 Enviando datos a {url}...")
    response = requests.post(url, json=payload)
    
    # Verificamos si la respuesta fue exitosa (Código 200)
    if response.status_code == 200:
        print("✅ ¡Éxito! La API respondió:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Error {response.status_code}:")
        print(response.text)

except Exception as e:
    print(f"❌ Error de conexión: {e}")
    print("Asegúrate de que uvicorn esté corriendo en otra terminal.")