# exportar_modelo_final.py
import os
import shutil

# === CONFIGURACIÓN ===
os.makedirs("export", exist_ok=True)

# Archivos esenciales para exportar
archivos_necesarios = [
    "models/modelo_final.keras",
    "models/scaler_final.pkl",
    "models/encoder_final.pkl"
]

print("🚀 Iniciando exportación del modelo final...\n")

# Copiar los archivos al directorio export/
for archivo in archivos_necesarios:
    if os.path.exists(archivo):
        shutil.copy(archivo, "export/")
        print(f"✅ Copiado: {archivo}")
    else:
        print(f"⚠️ No se encontró: {archivo}")

# Verificar resultado
contenido = os.listdir("export")
print("\n📦 Archivos exportados correctamente:")
for f in contenido:
    print(f"   - {f}")

print("\n🎯 Exportación completa: ahora tu modelo está listo para integración en otros sistemas.")
