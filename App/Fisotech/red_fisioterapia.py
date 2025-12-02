import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# ─── 1️⃣ Cargar datos de entrenamiento ─────────────────────────────
try:
    data = pd.read_csv("datos_movimiento.csv", header=None)
    data.columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    print(f"📂 Se cargaron {len(data)} registros para entrenamiento")
except:
    print("❌ No se encontró el archivo datos_movimiento.csv")
    exit()

# ─── 2️⃣ Simular etiquetas REALISTAS ─────────────────────────────
movimientos = ['reposo', 'movimiento', 'estiramiento']
labels = [movimientos[i % len(movimientos)] for i in range(len(data))]
data['etiqueta'] = labels

# ─── 3️⃣ Separar características y etiquetas ─────────────────────────────
X = data[['ax','ay','az','gx','gy','gz']]
y = pd.get_dummies(data['etiqueta'])  # convierte a one-hot

# ─── 4️⃣ Normalizar ─────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar scaler para usar en pruebas
joblib.dump(scaler, "scaler_fisioterapia.save")
print("✅ Scaler guardado como scaler_fisioterapia.save")

# ─── 5️⃣ Dividir en entrenamiento y prueba ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ─── 6️⃣ Crear red neuronal ─────────────────────────────
model = Sequential([
    Dense(16, input_dim=6, activation='relu'),
    Dense(12, activation='relu'),
    Dense(y.shape[1], activation='softmax')  # número de clases según y
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ─── 7️⃣ Entrenar ─────────────────────────────
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# ─── 8️⃣ Evaluar ─────────────────────────────
loss, acc = model.evaluate(X_test, y_test)
print(f"🎯 Precisión del modelo: {acc*100:.2f}%")

# ─── 9️⃣ Guardar modelo ─────────────────────────────
model.save("modelo_fisioterapia.h5")
print("✅ Modelo guardado como modelo_fisioterapia.h5")
