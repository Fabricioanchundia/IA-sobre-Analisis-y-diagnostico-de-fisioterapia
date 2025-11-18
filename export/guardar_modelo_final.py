# guardar_modelo_final.py
from tensorflow.keras.models import load_model
import os
import shutil

# Crear carpeta models si no existe
os.makedirs("models", exist_ok=True)

# Ruta del mejor modelo que ya entrenaste
modelo_origen = "models/modelo_combinado.keras"
modelo_destino = "models/modelo_final.keras"

# Verificar si el modelo base existe
if not os.path.exists(modelo_origen):
    raise FileNotFoundError("❌ No se encontró el archivo 'models/modelo_combinado.keras'. "
                            "Primero ejecuta entrenar_modelo.py.")

# Cargar y guardar versión final
modelo = load_model(modelo_origen)
modelo.save(modelo_destino)

print("✅ Modelo final guardado exitosamente como 'models/modelo_final.keras'")

# Copiar scaler y encoder también (opcional pero recomendado)
scaler_path = "models/scaler.pkl"
encoder_path = "models/encoder.pkl"

if os.path.exists(scaler_path):
    shutil.copy(scaler_path, "models/scaler_final.pkl")
if os.path.exists(encoder_path):
    shutil.copy(encoder_path, "models/encoder_final.pkl")

print("📦 Archivos auxiliares (scaler y encoder) también guardados como versión final.")
