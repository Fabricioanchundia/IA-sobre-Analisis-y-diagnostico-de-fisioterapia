import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import layers, models
import joblib
import os

# === CONFIGURACIÓN ===
DATA_PATH = "data/datos_sinteticos_v3.csv"
MODEL_RF_PATH = "models/modelo_rf_v22.pkl"
MODEL_NN_PATH = "models/modelo_nn_v22.h5"
ENCODER_PATH = "models/encoder_v22.pkl"
SCALER_PATH = "models/scaler_v22.pkl"

np.random.seed(42)

print("📂 Cargando dataset sintético extendido...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el archivo {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

# === VALIDACIÓN DE COLUMNAS ===
required_cols = ["ax", "ay", "az", "gx", "gy", "gz", "intensidad",
                 "magnitud_acc", "velocidad_ang", "energia_mov", "etiqueta"]
if not all(c in data.columns for c in required_cols):
    raise ValueError(f"❌ Faltan columnas requeridas. Se esperaban: {required_cols}")

print(f"✅ Dataset cargado correctamente: {len(data)} muestras")

# === PREPROCESAMIENTO ===
X = data.drop(columns=["etiqueta"])
y = data["etiqueta"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === DIVISIÓN TRAIN/TEST ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42
)

print(f"📊 Clases detectadas: {list(encoder.classes_)}")

# === MODELO DE RED NEURONAL ===
print("\n🧠 Entrenando red neuronal (extracción de características)...")

nn = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = nn.fit(
    X_train, y_train,
    epochs=80,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# === EXTRACCIÓN DE FEATURES INTERMEDIAS ===
extractor = models.Model(inputs=nn.inputs, outputs=nn.layers[-2].output)
features_train = extractor.predict(X_train)
features_test = extractor.predict(X_test)

# === RANDOM FOREST SOBRE FEATURES ===
print("\n🌲 Entrenando modelo Random Forest sobre representaciones neuronales...")
rf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42)
rf.fit(features_train, y_train)

# === EVALUACIÓN ===
preds = rf.predict(features_test)
acc = accuracy_score(y_test, preds)

print("\n📈 RESULTADOS DEL MODELO HÍBRIDO v22")
print(f"✅ Precisión global: {acc:.4f}")
print("\n📊 Clasificación por clase:")
print(classification_report(y_test, preds, target_names=encoder.classes_))
print("🧩 Matriz de confusión:")
print(confusion_matrix(y_test, preds))

# === GUARDAR MODELOS Y ARTEFACTOS ===
os.makedirs("export", exist_ok=True)
nn.save(MODEL_NN_PATH)
joblib.dump(rf, MODEL_RF_PATH)
joblib.dump(encoder, ENCODER_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n💾 Modelos y transformadores guardados exitosamente:")
print(f" - Red neuronal: {MODEL_NN_PATH}")
print(f" - Random Forest: {MODEL_RF_PATH}")
print(f" - Encoder: {ENCODER_PATH}")
print(f" - Scaler: {SCALER_PATH}")

print("\n🎉 Entrenamiento híbrido v22 completado correctamente.")
