import streamlit as st
import pandas as pd
import numpy as np
import joblib 

# --- Carga de Artefactos (Simulación de la API) ---
# Intenta cargar el modelo MLP y el scaler que fueron guardados en el Notebook.
try:
    mlp_clf = joblib.load('modelo_mlp_fraude.joblib')
    scaler = joblib.load('scaler_fraude.joblib')
except FileNotFoundError:
    st.error("Error: Los archivos del modelo (.joblib) no fueron encontrados. Asegúrate de que 'modelo_mlp_fraude.joblib' y 'scaler_fraude.joblib' estén en la misma carpeta.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Configuración de la página
st.set_page_config(page_title="FinSecure SpA - Dashboard", layout="wide")

# Título y Logo
st.title("🛡️ FinSecure SpA | Sistema de Detección de Fraude")
st.markdown("### Panel de Control para Analistas de Riesgo (Sistema de Soporte a la Decisión)")
st.markdown("---")

# --- BARRA LATERAL (Simulación de menú y control) ---
with st.sidebar:
    st.header("Configuración del Modelo")
    st.info("Modelo: Perceptrón Multicapa (MLP)")
    
    # Deslizador para ajustar el Umbral de Decisión (Aprendizaje Organizacional clave)
    umbral = st.slider("Umbral de Decisión (Threshold)", 0.0, 1.0, 0.20, help="Umbral ajustado a 0.20 para optimizar el Recall y maximizar la seguridad [cite: 283-284].")
    
    st.write(f"Estado de Sensibilidad: {'⚠️ ALERTA ALTA' if umbral < 0.3 else 'Norma Estándar'}")
    
    st.markdown("---")
    st.write("**KPIs de Referencia**")
    st.metric("Total de Transacciones/hr", "12,450")
    st.metric("Fraudes Detectados (Última hora)", "84", "+12%")

# --- CUERPO PRINCIPAL ---

# 1. Función de Inferencia (Usa el modelo y scaler cargados)
def simular_inferencia(data_list):
    # Definir nombres de features (Necesario para el Scaler, según Aprendizaje Técnico)
    # [cite_start]Se usan las 28 variables V1-V28 y Amount [cite: 153, 157]
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Crear DataFrame de 1 fila para escalar
    transaccion_df = pd.DataFrame([data_list], columns=feature_names)
    
    # Escalar los datos usando el objeto 'scaler' entrenado
    transaccion_escalada = scaler.transform(transaccion_df)
    
    # Predecir la probabilidad de fraude
    probabilidades = mlp_clf.predict_proba(transaccion_escalada)
    return probabilidades[0][1]

# Simulación de datos de ALTO RIESGO y BAJO RIESGO para demostrar el filtro de alertas
# Estos datos anonimizados (PCA) simulan transacciones con alto y bajo riesgo.
simulated_transactions = [
    {'ID': 'TRX-12345', 'Amount': 2500.00, 'Data': [-0.769, 1.342, -2.171, -0.151, -0.648, -0.973, -1.706, 0.313, -1.982, -3.158, 1.341, -3.299, 1.247, -6.393, -0.053, -3.258, -3.348, 0.217, -0.917, -0.102, -0.036, -0.753, -0.047, 0.358, -0.287, 0.476, 0.250, 0.250, 40.0]}, # Alto Riesgo (Fraude simulado)
    {'ID': 'TRX-67890', 'Amount': 12.50, 'Data': [-0.855, -0.100, -0.303, -0.329, 1.012, 0.411, 1.261, -0.197, 0.617, -0.816, 0.149, 0.311, -0.776, -1.767, -2.508, -0.068, 0.620, 0.178, 0.724, -0.424, -0.436, -0.465, 0.533, 0.133, -0.863, 0.071, -0.024, -0.076, 130.68]}, # Bajo Riesgo (No Fraude simulado)
    {'ID': 'TRX-11223', 'Amount': 3400.00, 'Data': [-0.99, 1.5, -2.5, -0.2, -0.7, -1.1, -1.8, 0.35, -2.1, -3.3, 1.4, -3.5, 1.3, -6.5, -0.1, -3.4, -3.5, 0.25, -0.95, -0.15, -0.05, -0.8, -0.05, 0.4, -0.3, 0.5, 0.3, 0.3, 50.0]}, # Alto Riesgo 2
    {'ID': 'TRX-44556', 'Amount': 150.00, 'Data': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 150.0]}, # Normal
    {'ID': 'TRX-77889', 'Amount': 10.00, 'Data': [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, 10.0]} # Normal
]

# Calcular el Risk Score real usando el modelo cargado (ejecutando la inferencia)
if 'mlp_clf' in locals():
    for tx in simulated_transactions:
        tx['Probabilidad Fraude (Score)'] = simular_inferencia(tx['Data'])

# Lógica de Negocio: Filtrar por Umbral ajustado
alertas = [tx for tx in simulated_transactions if tx['Probabilidad Fraude (Score)'] >= umbral]

# 2. Sección de Alertas Prioritarias (La interfaz del analista)
st.subheader("🚨 Cola de Alertas Prioritarias")
st.markdown(f"Mostrando transacciones con **Risk Score ≥ {umbral:.2f}**. [cite_start]Esto filtra el ruido y reduce la **fatiga de alertas**[cite: 183].")

if alertas:
    st.warning(f"Tienes {len(alertas)} transacciones de alto riesgo pendientes de revisión.")
    
    for index, row in enumerate(alertas):
        # Tarjeta de alerta con información y acciones
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
            
            with col1:
                st.subheader(f"ID: {row['ID']}")
            
            with col2:
                st.metric("Monto", f"${row['Amount']:.2f}", delta_color="off")
                st.info(f"Probabilidad Fraude: **{row['Probabilidad Fraude (Score)']:.2%}**")
            
            with col3:
                st.write("**Decisión del Modelo:**")
                st.error("Alerta Generada por MLP (Score sobre umbral)")
                # [cite_start]La explicación XAI se propone como mejora [cite: 230]
                st.caption("Factores: Combinación de V4, V10 y V14. **Datos anonimizados (PCA)** garantizan privacidad [cite: 235-237].")
                
            with col4:
                st.write("**Acción del Analista (Decisión Final):**")
                
                # [cite_start]Se refuerza que el humano tiene la autoridad final, mitigando el riesgo ético [cite: 227]
                if st.button(f"Bloquear Tarjeta y Fraude", key=f"btn_block_{row['ID']}"):
                    st.success("✅ Fraude confirmado y bloqueo ejecutado.")
                
                if st.button(f"Marcar como Falso Positivo (Liberar)", key=f"btn_ok_{row['ID']}"):
                    st.info("✔️ Transacción liberada. Analista corrigió FP.")
else:
    st.success("No hay alertas de fraude pendientes de revisión humana en este momento.")

# 3. Sección de Estadísticas (Resultados del Informe)
st.markdown("---")
st.subheader("📊 Resultados Validados del Modelo (Fase Piloto)")
st.caption("Métricas obtenidas sobre el Test Set, demostrando cumplimiento de Metas de Desempeño Técnico.")
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.metric(label="Recall (Sensibilidad)", value="84%", help="Identifica el 84% de los fraudes reales[cite: 288].")
with col_m2:
    st.metric(label="AUC-ROC", value="0.967", help="Muestra una robusta capacidad de separación entre clases[cite: 265].")
with col_m3:
    st.metric(label="Precisión Mínima", value="0.69", help="Valor aceptado para asegurar alertas de alta calidad[cite: 267].")