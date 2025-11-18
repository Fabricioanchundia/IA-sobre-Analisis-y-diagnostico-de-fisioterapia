import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

# === CONFIGURACIÓN ===
INPUT_PATH = "data/datos_balanceados_limpio.csv"
OUTPUT_PATH = "data/datos_sinteticos_v3_limpio.csv"
np.random.seed(42)

print("📂 Cargando dataset base...")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"No se encontró el archivo base: {INPUT_PATH}")

data = pd.read_csv(INPUT_PATH)

# === VALIDAR COLUMNAS REQUERIDAS ===
required_cols = ["ax", "ay", "az", "gx", "gy", "gz", "intensidad", "etiqueta"]
faltantes = [c for c in required_cols if c not in data.columns]

if faltantes:
    raise ValueError(f"❌ Faltan columnas requeridas: {faltantes}")

print(f"✅ Dataset base cargado correctamente con {len(data)} muestras.")
print(f"📊 Clases encontradas: {data['etiqueta'].unique().tolist()}")

# === FUNCIÓN PARA GENERAR VARIACIONES ===
def generar_variaciones(df, ruido_ax=0.05, ruido_gx=0.1, mult_intensidad=(0.9, 1.1)):
    """Genera nuevas muestras realistas aplicando ruido y rotaciones leves."""
    variado = df.copy()

    # Ruido en acelerómetro y giroscopio
    ruido_acel = np.random.normal(0, ruido_ax, size=(len(df), 3))
    ruido_gyro = np.random.normal(0, ruido_gx, size=(len(df), 3))

    variado[["ax", "ay", "az"]] += ruido_acel
    variado[["gx", "gy", "gz"]] += ruido_gyro

    # Rotación simulada (ligera mezcla de ejes X e Y)
    rot = np.random.uniform(-0.05, 0.05)
    ax_temp = variado["ax"].copy()
    variado["ax"] = ax_temp + rot * variado["ay"]
    variado["ay"] = variado["ay"] - rot * ax_temp

    # Variar la intensidad muscular un poco
    variado["intensidad"] *= np.random.uniform(mult_intensidad[0], mult_intensidad[1], size=len(df))

    return variado

# === BALANCEO Y GENERACIÓN ===
print("⚙️ Generando datos sintéticos balanceados...")

grupos = [g for _, g in data.groupby("etiqueta")]
max_len = max(len(g) for g in grupos)
data_final = []

for g in grupos:
    clase = g["etiqueta"].iloc[0]
    print(f"🧩 Procesando clase '{clase}' ({len(g)} muestras)...")

    # Aumentar hasta igualar al máximo
    while len(g) < max_len:
        extra = generar_variaciones(g)
        g = pd.concat([g, extra]).reset_index(drop=True)

    # Reducir si sobra
    g = resample(g, replace=False, n_samples=max_len, random_state=42)
    data_final.append(g)

# === COMBINAR Y MEZCLAR ===
df_sintetico = pd.concat(data_final).sample(frac=1, random_state=42).reset_index(drop=True)

# === CALCULAR FEATURES DERIVADAS ===
print("🧠 Calculando características derivadas...")

df_sintetico["magnitud_acc"] = np.sqrt(df_sintetico["ax"]**2 + df_sintetico["ay"]**2 + df_sintetico["az"]**2)
df_sintetico["velocidad_ang"] = np.sqrt(df_sintetico["gx"]**2 + df_sintetico["gy"]**2 + df_sintetico["gz"]**2)
df_sintetico["energia_mov"] = df_sintetico["magnitud_acc"] * df_sintetico["velocidad_ang"]

# === LIMPIEZA FINAL ===
df_sintetico = df_sintetico.loc[:, ~df_sintetico.columns.str.contains('^Unnamed')]
df_sintetico = df_sintetico.dropna().reset_index(drop=True)

# === GUARDAR ===
os.makedirs("data", exist_ok=True)
df_sintetico.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Dataset sintético v3 limpio generado correctamente.")
print(f"📁 Guardado en: {OUTPUT_PATH}")
print(f"📊 Total de muestras: {len(df_sintetico)}")
print("📑 Columnas finales:")
print(df_sintetico.columns.tolist())
