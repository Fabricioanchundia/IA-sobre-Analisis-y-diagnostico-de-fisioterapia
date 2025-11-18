import tkinter as tk
from tkinter import ttk
import threading
import serial
import requests
import joblib
import time
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

# === CONFIGURACIÓN ===
PORT = "COM3"
BAUDRATE = 9600
API_URL = "http://127.0.0.1:5000/predict"
scaler = joblib.load("models/scaler_v22.pkl")

arduino = None
is_running = False

# === HISTORIAL ===
hist_size = 50
intensity_history = deque(maxlen=hist_size)
confidence_history = deque(maxlen=hist_size)
movement_history = deque(maxlen=5)

# === INTERFAZ PRINCIPAL ===
root = tk.Tk()
root.title("🧠 FisioTech PRO — Dashboard Inteligente")
root.geometry("950x650")
root.configure(bg="#0a0f24")

# === ESTILOS ===
style = ttk.Style()
style.theme_use("clam")
style.configure("Verde.Horizontal.TProgressbar", troughcolor="#1e1e1e", background="lime")
style.configure("Amarillo.Horizontal.TProgressbar", troughcolor="#1e1e1e", background="yellow")
style.configure("Rojo.Horizontal.TProgressbar", troughcolor="#1e1e1e", background="red")

# === CABECERA ===
title = tk.Label(root, text="🧠 FisioTech PRO", font=("Segoe UI", 26, "bold"), fg="#00FFFF", bg="#0a0f24")
subtitle = tk.Label(root, text="Sistema de Monitoreo de Movimiento Fisioterapéutico en Tiempo Real",
                    font=("Segoe UI", 12), fg="#AAAAAA", bg="#0a0f24")
title.pack(pady=10)
subtitle.pack(pady=5)

# === FRAME PRINCIPAL ===
main_frame = tk.Frame(root, bg="#0a0f24")
main_frame.pack(pady=10, fill="both", expand=True)

# === PANEL IZQUIERDO ===
left_frame = tk.Frame(main_frame, bg="#0a0f24")
left_frame.pack(side="left", padx=20, fill="y")

mov_label = tk.Label(left_frame, text="🧩 Movimiento: ?", font=("Segoe UI", 18, "bold"), fg="white", bg="#0a0f24")
mov_label.pack(pady=10)

conf_label = tk.Label(left_frame, text="💪 Confianza: 0.000", font=("Segoe UI", 16), fg="#AAAAAA", bg="#0a0f24")
conf_label.pack(pady=5)

progress = ttk.Progressbar(left_frame, orient="horizontal", length=300, mode="determinate")
progress.pack(pady=10)
progress_val = tk.Label(left_frame, text="0%", font=("Segoe UI", 12), fg="#AAAAAA", bg="#0a0f24")
progress_val.pack()

status_label = tk.Label(left_frame, text="Esperando conexión...", fg="#AAAAAA", bg="#0a0f24", font=("Segoe UI", 10))
status_label.pack(pady=15)

# === BOTONES ===
btn_frame = tk.Frame(left_frame, bg="#0a0f24")
btn_frame.pack(pady=15)
btn_con = ttk.Button(btn_frame, text="Conectar Arduino", command=lambda: conectar_arduino())
btn_con.grid(row=0, column=0, padx=5)
btn_des = ttk.Button(btn_frame, text="Desconectar", command=lambda: desconectar_arduino())
btn_des.grid(row=0, column=1, padx=5)

# === PANEL DERECHO (GRÁFICAS) ===
right_frame = tk.Frame(main_frame, bg="#0a0f24")
right_frame.pack(side="right", padx=20, fill="both", expand=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.5))
fig.patch.set_facecolor("#0a0f24")
ax1.set_facecolor("#101830")
ax2.set_facecolor("#101830")

ax1.set_title("📈 Intensidad del Movimiento", color="#00FFFF")
ax1.set_ylim(0, 2)
ax1.set_xlim(0, hist_size)

ax2.set_title("💪 Nivel de Confianza del Modelo", color="#00FFFF")
ax2.set_ylim(0, 1)
ax2.set_xlim(0, hist_size)

line1, = ax1.plot([], [], color="#00FFAA", lw=2)
line2, = ax2.plot([], [], color="#FFFF00", lw=2)

canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)


# === FUNCIONES ===
def conectar_arduino():
    global arduino, is_running
    try:
        arduino = serial.Serial(PORT, BAUDRATE)
        time.sleep(2)
        status_label.config(text="✅ Conectado a Arduino", fg="lime")
        is_running = True
        threading.Thread(target=leer_datos, daemon=True).start()
        actualizar_graficas()
    except Exception as e:
        status_label.config(text=f"❌ Error: {e}", fg="red")


def desconectar_arduino():
    global is_running
    is_running = False
    if arduino and arduino.is_open:
        arduino.close()
        status_label.config(text="🔌 Desconectado", fg="orange")


def leer_datos():
    global is_running
    while is_running:
        try:
            line = arduino.readline().decode(errors="ignore").strip()
            if not line or "," not in line:
                continue
            valores = [float(x) for x in line.split(",")]
            if len(valores) < 7:
                continue

            ax, ay, az, gx, gy, gz, intensidad = valores
            magnitud_acc = math.sqrt(ax**2 + ay**2 + az**2)
            velocidad_ang = math.sqrt(gx**2 + gy**2 + gz**2)
            energia_mov = magnitud_acc * velocidad_ang

            features = [ax, ay, az, gx, gy, gz, intensidad, magnitud_acc, velocidad_ang, energia_mov]
            features_scaled = scaler.transform([features])[0].tolist()

            r = requests.post(API_URL, json={"features": features_scaled})
            data = r.json()
            pred = data.get("prediccion", "?").upper()
            conf = float(data.get("confianza", 0.0))

            actualizar_ui(pred, conf, intensidad)
            time.sleep(0.3)

        except Exception as e:
            status_label.config(text=f"⚠️ Error: {e}", fg="red")
            time.sleep(0.5)


def actualizar_ui(pred, conf, intensidad):
    mov_label.config(text=f"🧩 Movimiento: {pred}")
    conf_label.config(text=f"💪 Confianza: {conf:.3f}")

    porcentaje = int(conf * 100)
    progress["value"] = porcentaje
    progress_val.config(text=f"{porcentaje}%")

    if conf >= 0.8:
        progress.configure(style="Verde.Horizontal.TProgressbar")
    elif conf >= 0.5:
        progress.configure(style="Amarillo.Horizontal.TProgressbar")
    else:
        progress.configure(style="Rojo.Horizontal.TProgressbar")

    confidence_history.append(conf)
    intensity_history.append(intensidad)
    movement_history.append(pred)


def actualizar_graficas():
    if is_running:
        line1.set_data(range(len(intensity_history)), list(intensity_history))
        line2.set_data(range(len(confidence_history)), list(confidence_history))
        ax1.set_xlim(0, hist_size)
        ax2.set_xlim(0, hist_size)
        canvas.draw()
    root.after(500, actualizar_graficas)


# === CIERRE SEGURO ===
def cerrar_app():
    desconectar_arduino()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", cerrar_app)
root.mainloop()
