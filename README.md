# 📘 FisioTech PRO – Sistema Inteligente de Monitoreo Fisioterapéutico

**FisioTech PRO** es una plataforma híbrida (**IA + sensores**) que:

* Monitorea movimientos corporales en tiempo real
* Detecta patrones motores y niveles de intensidad
* Evalúa estabilidad y variabilidad del gesto
* Genera una **interpretación fisioterapéutica básica** con Machine Learning y un sistema experto de reglas

Integra:

* 🧠 Modelo de IA con clasificación y nivel de confianza
* ⚡ Sensor **MPU6050**
* 🔌 **Arduino** + comunicación serial
* 📊 Interfaz gráfica avanzada con Tkinter + Matplotlib
* 🩺 Diagnóstico fisioterapéutico básico
* 🎨 UI dark theme con logo de brillo suave

---

## 🚀 Características principales

### ✔️ Clasificación de movimiento en tiempo real

El sistema predice gestos como:

* TORSIÓN
* PIERNA
* BRAZO
* HOMBRO
* CAMINAR
* Reposo y otras variaciones

Se usa un modelo entrenado y escalado con `models/scaler_v22.pkl`.

---

### ✔️ Cálculo de métricas biomecánicas

A partir de datos del MPU6050:

* Magnitud total del acelerómetro
* Velocidad angular
* Energía del movimiento
* Intensidad instantánea
* Variabilidad del gesto

Estas métricas alimentan el sistema experto y la visualización.

---

### ✔️ Diagnóstico fisioterapéutico con reglas

Considera:

* Estabilidad del movimiento
* Variabilidad
* Tendencia
* Nivel de confianza del modelo

Clasifica el estado general en:

* 🟢 Movimiento estable
* 🟡 Movimiento irregular
* 🔴 Movimiento errático / señal de riesgo

Sugerencias del sistema:

* Reduzca la velocidad del movimiento
* Realice pausas activas
* Controle su postura
* Mantenga repeticiones isométricas suaves

> **Nota:** Interpretaciones académico‑educativas. No sustituyen valoración profesional de un fisioterapeuta o médico.

---

### ✔️ Interfaz gráfica profesional

* Logo animado con brillo suave
* Barras de progreso dinámicas en verde/amarillo/rojo
* Gráficas en tiempo real:

  * Intensidad de movimiento
  * Nivel de confianza del modelo

---

### ✔️ Integración directa con Arduino

Lectura serial en formato:

```
ax, ay, az, gx, gy, gz, intensidad
```

Compatible con:

* Arduino UNO / Nano
* MPU6050
* HC-05 Bluetooth — *próximamente*

---

## 🧩 Arquitectura del sistema

```
┌────────────────────────────┐
│      Arduino + MPU6050     │
│  Acelerómetro / Giroscopio │
└───────────────┬────────────┘
                │  Serial (USB)
                ▼
┌─────────────────────────────────┐
│         API Flask (IA)          │
│  /predict → MLPClassifier       │
│  + scaler_v22.pkl               │
└─────────────────────────────────┘
                │  JSON
                ▼
┌─────────────────────────────────┐
│      App Tkinter (Frontend)     │
│ - Clasificación                 │
│ - Gráficas en tiempo real       │
│ - Diagnóstico fisioterapéutico  │
└─────────────────────────────────┘
```

---

# 📂 Estructura recomendada del repositorio

```txt
IA-sobre-Analisis-y-diagnostico-de-fisioterapia/
│
├── assets/
│   └── logo_fisiotech.png
│
├── models/
│   └── scaler_v22.pkl
│
├── backend/
│   └── app.py
│
├── scripts/
│   └── fisiotech_pro_v29.py
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Dependencias del proyecto

Se necesita Python y librerías para IA, visualización, serial y backend.

## 📦 Dependencias de Python (requirements.txt)

Incluye al menos:

```
Flask==3.0.0
scikit-learn==1.3.2
numpy==1.26.4
pandas==2.1.4
matplotlib==3.8.2
Pillow==10.0.1
pyserial==3.5
joblib==1.3.2
```

### ¿Para qué sirve cada una?

| Librería         | Uso                                              |
| ---------------- | ------------------------------------------------ |
| **Flask**        | API backend para el modelo de IA                 |
| **scikit-learn** | Cargar modelo y scaler, hacer predicciones       |
| **numpy**        | Cálculos numéricos con señales                   |
| **pandas**       | Estructura de datos, preparación de input        |
| **matplotlib**   | Gráficas en tiempo real en GUI                   |
| **Pillow**       | Efecto de brillo, carga y manipulación de imagen |
| **pyserial**     | Comunicación serial con Arduino                  |
| **joblib**       | Cargar archivos `.pkl` del modelo y scaler       |

> Tkinter se distribuye normalmente con Python y sirve como GUI básica y potente. La documentación oficial indica que Tkinter es la interfaz estándar de Python para Tcl/Tk en la mayoría de plataformas, confirmando así su disponibilidad en instalaciones comunes de Python. ([Python documentation][1])

---

# 🔧 Dependencias externas necesarias

### 🟦 Arduino IDE

Para programar el Arduino y enviar datos.
Sitio oficial: [https://www.arduino.cc/en/software](https://www.arduino.cc/en/software)

### 🟩 Librerías para MPU6050 en Arduino

Desde **Arduino IDE → Library Manager** instalar:

* MPU6050
* Adafruit Unified Sensor
* Wire

### 🟨 Drivers USB para algunas placas

Si usas Arduino Nano con chip CH340, instala el driver correspondiente para que el puerto serial funcione en Windows.

---

# ⚙️ Instalación paso a paso

## 1️⃣ Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
```

Activar:

* **Windows**:

  ```bash
  venv\Scripts\activate
  ```
* **Linux/Mac**:

  ```bash
  source venv/bin/activate
  ```

## 2️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

Si ocurre algún error, instalar manualmente:

```bash
pip install flask scikit-learn numpy pandas matplotlib pillow pyserial joblib
```

---

# 🚀 Cómo ejecutar el proyecto

## 🔸 Paso 1: Configurar y ejecutar Arduino

1. Conecta el MPU6050 al Arduino.
2. Sube el código del Arduino que lee el sensor y envía por serial.
3. Usa **baudrate 9600** en el Arduino y, luego, en el script Python.
4. Verifica que el monitor serial muestre líneas con datos tipo:

   ```
   ax,ay,az,gx,gy,gz,intensidad
   ```

## 🔸 Paso 2: Ejecutar la API Flask (IA)

En una terminal:

```bash
cd backend
python app.py
```

Esto levantará el servidor (por ejemplo) en:

```
http://127.0.0.1:5000/predict
```

## 🔸 Paso 3: Ejecutar la interfaz principal

En otra terminal, con el entorno activado:

```bash
cd scripts
python fisiotech_pro_v29.py
```

La interfaz:

* Lee datos del Arduino en tiempo real
* Envía los datos al backend para predecir el movimiento
* Muestra el tipo de movimiento, el nivel de confianza
* Grafica intensidad y confianza
* Muestra diagnóstico y recomendaciones del sistema experto

---

# 📉 Roadmap — Próximas mejoras

* [ ] Integración HC-05 para Bluetooth real
* [ ] Exportar reportes en PDF para usuarios o terapeutas
* [ ] Detección automática de repeticiones
* [ ] Control postural más avanzado
* [ ] Versión móvil con Flutter o React Native
* [ ] Dashboard web con monitoreo remoto para terapeutas

---

# 👥 Autores

**Fabricio Anchundia Mero**
Estudiante de Ingeniería en Software – PUCE Manabí

**John Steven López Vélez**
Estudiante de Ingeniería en Software – PUCE Manabí

**Ariel Gonzalo Moreira Macías**
Estudiante de Ingeniería en Software – PUCE Manabí

---

# ⭐ Si te gustó este proyecto

Dale una estrella ⭐ en GitHub y ayuda a aumentar su visibilidad.

