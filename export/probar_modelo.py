import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# === CONFIGURACIÓN ===
os.makedirs("resultados", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# === DETECTAR MODELO COMBINADO ===
modelo_path = "models/modelo_combinado.keras"
if not os.path.exists(modelo_path):
    raise FileNotFoundError("❌ No se encontró 'models/modelo_combinado.keras'. Entrénalo antes de probarlo.")
print("🔍 Usando modelo combinado (reales + aumentados).")

# === CARGAR MODELO Y PREPROCESADORES ===
model = load_model(modelo_path)
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

# === CARGAR DATOS REALES Y AUMENTADOS ===
data_reales = pd.read_csv("data/datos_reales.csv")
data_aumentados = pd.read_csv("data/datos_aumentados.csv")

# === UNIR AMBOS ===
data_total = pd.concat([data_reales, data_aumentados], ignore_index=True)
print(f"✅ Datos combinados cargados: {len(data_total)} filas totales.")

# === VERIFICAR COLUMNAS ===
columnas = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'intensidad', 'etiqueta']
faltantes = [c for c in columnas if c not in data_total.columns]
if faltantes:
    raise ValueError(f"❌ Faltan columnas necesarias en los datos: {faltantes}")

# === PREPARAR ===
X = data_total[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'intensidad']].values
y_true = data_total['etiqueta'].values
y_true_encoded = encoder.transform(y_true)

# === ESCALAR ===
X_scaled = scaler.transform(X)

# === PREDICCIONES ===
y_pred_probs = model.predict(X_scaled)
y_pred_encoded = np.argmax(y_pred_probs, axis=1)
y_pred = encoder.inverse_transform(y_pred_encoded)

# === MÉTRICAS ===
precision = accuracy_score(y_true_encoded, y_pred_encoded)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"\n🎯 Precisión general del modelo combinado: {precision * 100:.2f}%")

# === REPORTE COMPLETO ===
reporte_texto = classification_report(y_true_encoded, y_pred_encoded, target_names=encoder.classes_)
print("\n📋 Reporte de Clasificación:\n", reporte_texto)

# Guardar reporte en TXT
with open(f"resultados/reporte_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write("=== REPORTE DE CLASIFICACIÓN ===\n")
    f.write(reporte_texto)
print(f"📝 Reporte guardado en 'resultados/reporte_{timestamp}.txt'")

# === GUARDAR RESULTADOS EN CSV ===
resultados = pd.DataFrame({"Etiqueta Real": y_true, "Predicción": y_pred})
csv_path = f"resultados/predicciones_combinadas_{timestamp}.csv"
resultados.to_csv(csv_path, index=False)
print(f"💾 Predicciones guardadas en '{csv_path}'")

# === 1️⃣ COMPARACIÓN DE ETIQUETAS ===
plt.figure(figsize=(12,5))
plt.plot(y_true, 'bo-', label='Real', alpha=0.6)
plt.plot(y_pred, 'r.--', label='Predicha', alpha=0.6)
plt.legend()
plt.title("Comparación de Etiquetas Reales vs Predichas")
plt.xticks(rotation=45)
plt.tight_layout()
graf1_path = f"resultados/comparacion_{timestamp}.png"
plt.savefig(graf1_path)
plt.close()
print(f"📸 Gráfico comparativo guardado en '{graf1_path}'")

# === 2️⃣ MATRIZ DE CONFUSIÓN ===
cm = confusion_matrix(y_true_encoded, y_pred_encoded)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
plt.figure(figsize=(8,6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusión - Modelo Combinado")
plt.tight_layout()
graf2_path = f"resultados/matriz_confusion_{timestamp}.png"
plt.savefig(graf2_path)
plt.close()
print(f"🧩 Matriz de confusión guardada en '{graf2_path}'")

# === 3️⃣ HEATMAP DE PROBABILIDADES ===
plt.figure(figsize=(10,6))
sns.heatmap(y_pred_probs, cmap="magma", cbar=True)
plt.title("Mapa de Calor de Probabilidades")
plt.xlabel("Clases")
plt.ylabel("Muestras")
plt.tight_layout()
graf3_path = f"resultados/heatmap_{timestamp}.png"
plt.savefig(graf3_path)
plt.close()
print(f"🔥 Mapa de calor guardado en '{graf3_path}'")

# === 4️⃣ RENDIMIENTO POR CLASE ===
reporte = classification_report(y_true_encoded, y_pred_encoded, target_names=encoder.classes_, output_dict=True)
rendimiento_por_clase = pd.DataFrame(reporte).T
rendimiento_por_clase = rendimiento_por_clase.drop(["accuracy"], errors="ignore")

plt.figure(figsize=(10,6))
sns.barplot(
    x=rendimiento_por_clase.index,
    y=rendimiento_por_clase["f1-score"],
    palette="viridis"
)
plt.title("📊 Rendimiento por Clase (F1-Score)")
plt.ylabel("F1-Score")
plt.xlabel("Clase")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
graf4_path = f"resultados/rendimiento_{timestamp}.png"
plt.savefig(graf4_path)
plt.close()
print(f"📈 Gráfico de rendimiento por clase guardado en '{graf4_path}'")

print("\n✅ Todo listo amor 💙 — informe, gráficos y métricas generados correctamente.")
