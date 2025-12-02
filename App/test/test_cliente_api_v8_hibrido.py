import requests
import pandas as pd
import numpy as np
import os

API_URL = "http://127.0.0.1:5000/predict"

print("🧠 Probando el modelo híbrido... Espere unos segundos.\n")

# === CARGAR EL DATASET ===
DATA_PATH = "data/datos_sinteticos_v3_limpio.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ No se encontró el archivo datos_sinteticos_v3_limpio.csv")

df = pd.read_csv(DATA_PATH)

# === SELECCIONAR COLUMNAS Y CLASES ===
features = ["ax","ay","az","gx","gy","gz","intensidad","magnitud_acc","velocidad_ang","energia_mov"]
etiquetas = df["etiqueta"].unique().tolist()

# Tomar una muestra de cada clase
muestras = []
for clase in etiquetas:
    subset = df[df["etiqueta"] == clase].head(1)  # una por clase
    if not subset.empty:
        muestras.append(subset)

df_test = pd.concat(muestras).reset_index(drop=True)

# === EVALUAR ===
resultados = []
for i, fila in df_test.iterrows():
    esperado = fila["etiqueta"]
    payload = {"features": fila[features].tolist()}

    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        if r.status_code == 200:
            pred = r.json()
            resultados.append({
                "esperado": esperado,
                "predicho": pred["prediccion"],
                "confianza": pred["confianza"]
            })
            print(f"{esperado.upper():10} → {pred['prediccion'].upper():10} ({pred['confianza']:.3f})")
        else:
            print(f"❌ Error con clase {esperado}: {r.text}")
    except Exception as e:
        print(f"❌ Error al conectar con API: {e}")

# === RESULTADOS ===
if len(resultados) > 0:
    df_res = pd.DataFrame(resultados)
    aciertos = (df_res["esperado"] == df_res["predicho"]).sum()
    total = len(df_res)
    promedio_conf = np.mean(df_res["confianza"])

    print("\n📊 RESULTADOS DEL MODELO HÍBRIDO v22")
    print("Movimiento Esperado  Predicción del Modelo  Confianza")
    print("-"*55)
    for _, row in df_res.iterrows():
        print(f"{row['esperado'].capitalize():10}   {row['predicho'].capitalize():10}   {row['confianza']:.3f}")

    print(f"\n✅ Exactitud: {aciertos}/{total} ({aciertos/total:.2%})")
    print(f"💪 Promedio de confianza global: {promedio_conf:.3f}")
else:
    print("\n⚠️ No se recibieron resultados.")
