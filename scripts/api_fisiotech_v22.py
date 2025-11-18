import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib, os
from tensorflow.keras.models import load_model, Model

# === RUTAS ===
MODEL_NN_PATH = "models/modelo_nn_v22.h5"
MODEL_RF_PATH = "models/modelo_rf_v22.pkl"
SCALER_PATH = "models/scaler_v22.pkl"
ENCODER_PATH = "models/encoder_v22.pkl"

# === NÚMERO DE FEATURES ORIGINALES ===
FEATURE_COUNT = 10  # ax ay az gx gy gz intensidad mag_acc vel_ang energia_mov

print("🚀 Cargando modelos híbridos v22...")

# === CARGA DE MODELOS ===
if not all(os.path.exists(p) for p in [MODEL_NN_PATH, MODEL_RF_PATH, SCALER_PATH, ENCODER_PATH]):
    raise FileNotFoundError("❌ No se encontraron todos los modelos entrenados. Ejecuta el reentrenamiento v22 primero.")

nn = load_model(MODEL_NN_PATH)
rf = joblib.load(MODEL_RF_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# ✅ EXTRAER CAPA INTERMEDIA (64 FEATURES)
extractor = Model(inputs=nn.inputs, outputs=nn.layers[-2].output)

print("✅ Modelos cargados correctamente.")

# === FLASK API ===
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Formato inválido. Se esperaba {'features': [valores...]}"})
        
        features = np.array(data["features"]).reshape(1, -1)

        # ✅ Validar tamaño correcto
        if features.shape[1] != FEATURE_COUNT:
            return jsonify({"error": f"X has {features.shape[1]} features, but the API expects {FEATURE_COUNT}."})

        # ✅ Escalar features
        X_scaled = scaler.transform(features)

        # ✅ Pasar por el extractor (produce 64 features)
        deep_features = extractor.predict(X_scaled)

        # ✅ Predicción RandomForest usando las 64 deep features
        pred_rf = rf.predict(deep_features)
        probs = rf.predict_proba(deep_features)

        label = encoder.inverse_transform(pred_rf)[0]
        confianza = float(np.max(probs))

        return jsonify({
            "prediccion": label,
            "confianza": round(confianza, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API FisioTech v22 funcionando correctamente ✅"})


if __name__ == "__main__":
    print("✅ Servidor API iniciado en http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
