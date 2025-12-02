import pandas as pd
import os

INPUT_PATH = "data/datos_balanceados.csv"
OUTPUT_PATH = "data/datos_balanceados_limpio.csv"

print("🧹 Limpiando profundamente el archivo base...")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo base: {INPUT_PATH}")

# === 1️⃣ Leer el CSV sin intentar usar encabezados corruptos ===
data = pd.read_csv(INPUT_PATH, header=0, low_memory=False)

# === 2️⃣ Eliminar columnas con nombres numéricos o NaN ===
data = data.loc[:, ~data.columns.astype(str).str.match("^-?\\d+$")]
data = data.loc[:, ~data.columns.str.contains("Unnamed", case=False, na=False)]

# === 3️⃣ Mantener solo las válidas esperadas ===
validas = ["ax", "ay", "az", "gx", "gy", "gz", "intensidad", "etiqueta"]
cols_encontradas = [c for c in validas if c in data.columns]

if len(cols_encontradas) < len(validas):
    print(f"⚠️ Algunas columnas faltan: {set(validas) - set(cols_encontradas)}")

data = data[cols_encontradas].dropna().reset_index(drop=True)

# === 4️⃣ Filtrar filas vacías y etiquetas inválidas ===
data = data[data["etiqueta"].astype(str).str.strip() != ""]
data = data[data["etiqueta"].notna()]

# === 5️⃣ Mostrar resumen ===
print(f"✅ Archivo limpio con {len(data)} filas válidas y columnas {list(data.columns)}")

# === 6️⃣ Guardar ===
os.makedirs("data", exist_ok=True)
data.to_csv(OUTPUT_PATH, index=False)

print(f"💾 Guardado en: {OUTPUT_PATH}")
