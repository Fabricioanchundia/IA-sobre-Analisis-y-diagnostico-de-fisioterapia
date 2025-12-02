import serial
import requests
import time
import math

# Configuración del puerto serial (ajusta si no es COM3)
PORT = "COM3"
BAUD_RATE = 9600
API_URL = "http://127.0.0.1:5000/predict"

print("🚀 Iniciando lectura del Arduino (modo simulación)...")
print("📡 Conectando al puerto", PORT)

# Intentar conexión serial
try:
    arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ Conectado correctamente al Arduino.\n")
except Exception as e:
    print("❌ Error al conectar con Arduino:", e)
    exit()

while True:
    try:
        # Leer línea desde Arduino
        linea = arduino.readline().decode('utf-8').strip()

        if not linea or "," not in linea:
            continue

        # Separar los valores
        valores = [float(x) for x in linea.split(",")]

        if len(valores) != 7:
            continue

        ax, ay, az, gx, gy, gz, intensidad = valores

        # Calcular features derivadas
        magnitud_acc = math.sqrt(ax**2 + ay**2 + az**2)
        velocidad_ang = math.sqrt(gx**2 + gy**2 + gz**2)
        energia_mov = magnitud_acc * velocidad_ang

        features = [ax, ay, az, gx, gy, gz, intensidad, magnitud_acc, velocidad_ang, energia_mov]

        # Enviar a la API IA v22
        response = requests.post(API_URL, json={"features": features})
        data = response.json()

        pred = data.get("prediccion", "?")
        conf = data.get("confianza", 0.0)

        print(f"🧩 Movimiento: {pred:<10} | Confianza: {conf:.3f}")

    except KeyboardInterrupt:
        print("\n🛑 Lectura detenida por el usuario.")
        break
    except Exception as e:
        print("⚠️ Error en lectura:", e)
        time.sleep(0.5)
