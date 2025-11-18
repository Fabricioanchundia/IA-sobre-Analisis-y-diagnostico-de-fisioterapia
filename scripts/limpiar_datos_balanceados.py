import pandas as pd
import os

INPUT_PATH = "data/datos_balanceados.csv"
OUTPUT_PATH = "data/datos_balanceados_limpio.csv"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_PATH}")

print("🧹 Limpiando archivo base...")

# Cargar CSV completo
data = pd.read_csv(INPUT_PATH)

# Eliminar columnas con nombres numéricos o corruptos
data = data.loc[:, ~data.columns.str.match("^-?\\d+$")]
data = data.loc[:, ~data.columns.str.contains("Unnamed", case=False, na=False)]

# Mantener solo columnas válidas
validas = ["ax", "ay", "az", "gx", "gy", "gz", "intensidad", "etiqueta"]
faltan = [c for c in validas if c not in data.columns]

if faltan:
    raise ValueError(f"❌ Faltan columnas necesarias: {faltan}")

data = data[validas].dropna().reset_index(drop=True)

print(f"✅ Archivo limpio con {len(data)} filas y columnas: {list(data.columns)}")

# Guardar limpio
os.makedirs("data", exist_ok=True)
data.to_csv(OUTPUT_PATH, index=False)

print(f"💾 Guardado en: {OUTPUT_PATH}")
