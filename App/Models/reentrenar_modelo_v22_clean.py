import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import joblib, os, sys

DATA_PATH = "data/datos_sinteticos_v3_limpio.csv"
os.makedirs("export", exist_ok=True)

print("📂 Cargando dataset sintético limpio...")

if not os.path.exists(DATA_PATH):
    sys.exit(f"❌ No se encontró el archivo: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
data = data.loc[:, ~data.columns.str.contains('^Unnamed', case=False)]
data = data.dropna()

features = [
    "ax","ay","az","gx","gy","gz",
    "intensidad","magnitud_acc","velocidad_ang","energia_mov"
]

if any(f not in data.columns for f in features):
    sys.exit("❌ Faltan columnas en el dataset.")

X = data[features].values
y = data["etiqueta"].values

print(f"✅ Dataset con {X.shape[0]} muestras y {X.shape[1]} features")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print(f"📊 Clases detectadas: {list(encoder.classes_)}")

# === RED NEURONAL ===
print("🧠 Entrenando red neuronal...")
nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation="softmax")
])

nn.compile(optimizer=Adam(learning_rate=0.001),
           loss="sparse_categorical_crossentropy",
           metrics=["accuracy"])

nn.fit(X_train, y_train, validation_data=(X_test, y_test),
       epochs=60, batch_size=16, verbose=1)

# === FEATURES INTERMEDIAS + RF ===
extractor = Model(inputs=nn.inputs, outputs=nn.layers[-2].output)
X_train_feat = extractor.predict(X_train)
X_test_feat = extractor.predict(X_test)

rf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42)
rf.fit(X_train_feat, y_train)

preds = rf.predict(X_test_feat)
print("\n📈 RESULTADOS DEL MODELO HÍBRIDO LIMPIO v22")
print(classification_report(y_test, preds, target_names=encoder.classes_))
print(confusion_matrix(y_test, preds))

# === GUARDAR MODELOS ===
nn.save("export/modelo_nn_v22.h5")
joblib.dump(rf, "export/modelo_rf_v22.pkl")
joblib.dump(scaler, "export/scaler_v22.pkl")
joblib.dump(encoder, "export/encoder_v22.pkl")

print("\n🎉 Entrenamiento híbrido v22 limpio completado correctamente.")
