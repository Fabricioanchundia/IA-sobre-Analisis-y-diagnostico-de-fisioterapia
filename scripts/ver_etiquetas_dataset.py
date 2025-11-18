import pandas as pd

data = pd.read_csv("data/datos_reales.csv")
print("=== Distribución de clases en datos_reales.csv ===")
print(data['etiqueta'].value_counts())

data2 = pd.read_csv("data/datos_aumentados.csv")
print("\n=== Distribución de clases en datos_aumentados.csv ===")
print(data2['etiqueta'].value_counts())
