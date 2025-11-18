import tkinter as tk
from tkinter import ttk
import threading
import serial
import requests
import joblib
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from PIL import Image, ImageTk, ImageEnhance

# === CONFIGURACIÓN GENERAL ===
PORT = "COM3"
BAUDRATE = 9600
API_URL = "http://127.0.0.1:5000/predict"
scaler = joblib.load("models/scaler_v22.pkl")

arduino = None
is_running = False
intensity_history = deque(maxlen=50)
confidence_history = deque(maxlen=50)

# === INTERFAZ PRINCIPAL ===
root = tk.Tk()
root.title("FisioTech PRO v2.9 — Sistema Inteligente de Monitoreo Fisioterapéutico")
root.geometry("1200x780")
root.configure(bg="#0a0f24")

# === LOGO CON EFECTO DE BRILLO ===
logo_path = "assets/logo_fisiotech.png"
base_logo = Image.open(logo_path).convert("RGBA")
logo_size = 140
logo_img = base_logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
logo_tk = ImageTk.PhotoImage(logo_img)

logo_frame = tk.Frame(root, bg="#0a0f24")
logo_frame.pack(pady=(10, 0))

logo_label = tk.Label(logo_frame, image=logo_tk, bg="#0a0f24")
logo_label.pack()

titulo_label = tk.Label(logo_frame, text="FisioTech PRO", font=("Segoe UI Semibold", 24),
                        fg="#00e5ff", bg="#0a0f24")
titulo_label.pack()

subtitulo_label = tk.Label(logo_frame,
                           text="Sistema Inteligente de Monitoreo Fisioterapéutico",
                           font=("Segoe UI", 11), fg="#cccccc", bg="#0a0f24")
subtitulo_label.pack(pady=(2, 15))

# === ANIMACIÓN DEL LOGO (solo brillo) ===
pulse_state = {"brightness": 1.0, "light_up": True}
def animar_logo():
    try:
        b = pulse_state["brightness"]
        l = pulse_state["light_up"]

        b += 0.05 if l else -0.05
        if b >= 1.6:
            pulse_state["light_up"] = False
        if b <= 0.6:
            pulse_state["light_up"] = True
        pulse_state["brightness"] = b

        enhancer = ImageEnhance.Brightness(base_logo)
        bright_img = enhancer.enhance(b)
        resized = bright_img.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
        logo_tk2 = ImageTk.PhotoImage(resized)
        logo_label.configure(image=logo_tk2)
        logo_label.image = logo_tk2
        root.after(120, animar_logo)
    except Exception:
        pass

animar_logo()

# === CONTENEDOR PRINCIPAL ===
content = tk.Frame(root, bg="#0a0f24")
content.pack(fill="both", expand=True, padx=30, pady=15)

# === PANEL IZQUIERDO ===
left = tk.Frame(content, bg="#0a0f24", highlightbackground="#00e5ff", highlightthickness=1)
left.pack(side="left", fill="y", padx=20, ipadx=15, ipady=10)

mov_label = tk.Label(left, text="Movimiento: ?", font=("Segoe UI", 22, "bold"),
                     fg="white", bg="#0a0f24")
mov_label.pack(pady=(20, 5))

conf_label = tk.Label(left, text="Confianza: 0.000", font=("Segoe UI", 13),
                      fg="#AAAAAA", bg="#0a0f24")
conf_label.pack()

style = ttk.Style()
style.theme_use("clam")
style.configure("Verde.Horizontal.TProgressbar", troughcolor="#1e1e1e", background="#00FF88")
style.configure("Amarillo.Horizontal.TProgressbar", troughcolor="#1e1e1e", background="#FFD700")
style.configure("Rojo.Horizontal.TProgressbar", troughcolor="#1e1e1e", background="#FF4040")

progress = ttk.Progressbar(left, orient="horizontal", length=380, mode="determinate")
progress.pack(pady=(15, 8))
progress_val = tk.Label(left, text="0%", font=("Segoe UI", 12), fg="#AAAAAA", bg="#0a0f24")
progress_val.pack()

status_label = tk.Label(left, text="⏳ Esperando conexión...", fg="#AAAAAA",
                        bg="#0a0f24", font=("Segoe UI", 11))
status_label.pack(pady=10)

btn_frame = tk.Frame(left, bg="#0a0f24")
btn_frame.pack(pady=20)
ttk.Button(btn_frame, text="Conectar Arduino", command=lambda: conectar_arduino()).grid(row=0, column=0, padx=10)
ttk.Button(btn_frame, text="Desconectar", command=lambda: desconectar_arduino()).grid(row=0, column=1, padx=10)

# === DIAGNÓSTICO CLÍNICO ===
diag_title = tk.Label(left, text="Interpretación Fisioterapéutica:",
                      font=("Segoe UI Semibold", 12), fg="#00e5ff", bg="#0a0f24")
diag_title.pack(pady=(15, 5), anchor="w")

diag_status = tk.Label(left, text="—", font=("Segoe UI", 11, "bold"),
                       fg="#FFD700", bg="#0a0f24")
diag_status.pack(anchor="w")

diag_msg = tk.Label(left, text="—", font=("Segoe UI", 10),
                    fg="#FFD700", bg="#0a0f24", justify="left", wraplength=360)
diag_msg.pack(anchor="w")

trat_title = tk.Label(left, text="Tratamiento sugerido:",
                      font=("Segoe UI Semibold", 12), fg="#00e5ff", bg="#0a0f24")
trat_title.pack(pady=(10, 3), anchor="w")

trat_msg = tk.Label(left, text="—", font=("Segoe UI", 10),
                    fg="#AAAAAA", bg="#0a0f24", justify="left", wraplength=360)
trat_msg.pack(anchor="w")

# === PANEL DERECHO (Gráficas) ===
right = tk.Frame(content, bg="#0a0f24")
right.pack(side="right", fill="both", expand=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6))
fig.subplots_adjust(hspace=0.45, top=0.92, bottom=0.12)
fig.patch.set_facecolor("#0a0f24")

for ax in (ax1, ax2):
    ax.set_facecolor("#101830")
    ax.grid(True, color="#1E2A45", linestyle="--", linewidth=0.6)
    ax.tick_params(axis='x', colors="#B0C4DE", labelsize=9)
    ax.tick_params(axis='y', colors="#B0C4DE", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1E2A45")

ax1.set_title("Intensidad del Movimiento (Acelerómetro)", color="#00FFFF", pad=12, fontsize=11)
ax1.set_ylim(0, 2)
ax1.set_xlim(0, 50)
ax1.set_ylabel("Intensidad", color="white", labelpad=8)

ax2.set_title("Nivel de Confianza del Modelo", color="#00FFFF", pad=12, fontsize=11)
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 50)
ax2.set_ylabel("Confianza", color="white", labelpad=8)
ax2.set_xlabel("Tiempo", color="white", labelpad=10)

line1, = ax1.plot([], [], color="#00FFAA", lw=2)
line2, = ax2.plot([], [], color="#FFD700", lw=2)

canvas = FigureCanvasTkAgg(fig, master=right)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

# === FUNCIONES PRINCIPALES ===
def conectar_arduino():
    global arduino, is_running
    try:
        arduino = serial.Serial(PORT, BAUDRATE)
        time.sleep(2)
        status_label.config(text="🟢 Conectado a Arduino.", fg="lime")
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

            parts = line.split(",")
            numeric_vals = []
            for val in parts[:7]:
                try:
                    numeric_vals.append(float(val))
                except ValueError:
                    continue

            if len(numeric_vals) < 7:
                continue

            ax, ay, az, gx, gy, gz, intensidad = numeric_vals
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
        except Exception:
            time.sleep(0.5)

def evaluar_estabilidad(int_hist: deque, conf_hist: deque):
    if len(int_hist) < 10:
        return {
            "titulo": "Analizando señal…",
            "mensaje": "Reuniendo datos suficientes para evaluar estabilidad.",
            "color": "#AAAAAA",
            "tratamiento": "Espere unos segundos hasta tener más lecturas."
        }

    intens = np.array(list(int_hist))
    confs = np.array(list(conf_hist)) if len(conf_hist) else np.array([0.0])
    std_int = np.std(intens)
    conf_med = np.mean(confs)

    if conf_med >= 0.75 and std_int <= 0.05:
        return {"titulo": "🟢 Movimiento estable.",
                "mensaje": "Patrón motor dentro del rango fisiológico.",
                "color": "#00FF88",
                "tratamiento": "Mantenga el ritmo y siga con el ejercicio controlado."}
    elif (0.5 <= conf_med < 0.75) or (0.05 < std_int <= 0.12):
        return {"titulo": "⚠️ Movimiento irregular.",
                "mensaje": "Posible falta de control o ligera desviación postural.",
                "color": "#FFD700",
                "tratamiento": "Reduzca la velocidad, haga pausas activas y controle su postura."}
    else:
        return {"titulo": "🔴 Movimiento riesgoso.",
                "mensaje": "Desviación significativa o baja confianza del sensor.",
                "color": "#FF4040",
                "tratamiento": "Detenga el ejercicio y revise la técnica o ajuste del sensor."}

def actualizar_diagnostico():
    info = evaluar_estabilidad(intensity_history, confidence_history)
    diag_status.config(text=info["titulo"], fg=info["color"])
    diag_msg.config(text=info["mensaje"], fg=info["color"])
    trat_msg.config(text=info["tratamiento"], fg=info["color"])

def actualizar_ui(pred, conf, intensidad):
    mov_label.config(text=f"Movimiento: {pred}")
    conf_label.config(text=f"Confianza: {conf:.3f}")
    progress["value"] = int(conf * 100)
    progress_val.config(text=f"{int(conf * 100)}%")

    if conf >= 0.8:
        progress.configure(style="Verde.Horizontal.TProgressbar")
    elif conf >= 0.5:
        progress.configure(style="Amarillo.Horizontal.TProgressbar")
    else:
        progress.configure(style="Rojo.Horizontal.TProgressbar")

    confidence_history.append(conf)
    intensity_history.append(intensidad)
    actualizar_diagnostico()

def actualizar_graficas():
    if is_running:
        line1.set_data(range(len(intensity_history)), list(intensity_history))
        line2.set_data(range(len(confidence_history)), list(confidence_history))
        canvas.draw()
    root.after(500, actualizar_graficas)

def cerrar():
    desconectar_arduino()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", cerrar)
root.mainloop()
