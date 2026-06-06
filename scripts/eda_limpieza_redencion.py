# ============================================================
# EDA Y LIMPIEZA
# Proyecto: Predicción de Redención de Puntos
# Autor: Grupo 3
# Entorno: Python / Pandas / PySpark preparado para Docker
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. RUTAS DEL PROYECTO
# ------------------------------------------------------------

RUTA_EXCEL = "data/Tabla_tranx_1mar_31mar.xlsx"
RUTA_CSV_LIMPIO = "data/transacciones_redencion_limpio.csv"
RUTA_REPORTE_EDA = "outputs/reporte_eda.txt"
RUTA_GRAFICAS = "graficas"

os.makedirs("outputs", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

# ------------------------------------------------------------
# 2. CARGA DEL DATASET
# ------------------------------------------------------------

print("Cargando dataset...")

df = pd.read_excel(RUTA_EXCEL)

print("Dataset cargado correctamente")
print("Filas y columnas:", df.shape)

# ------------------------------------------------------------
# 3. EXPLORACIÓN INICIAL
# ------------------------------------------------------------

print("\nCOLUMNAS DEL DATASET")
print(df.columns.tolist())

print("\nPRIMERAS FILAS")
print(df.head())

print("\nTIPOS DE DATOS")
print(df.dtypes)

print("\nVALORES NULOS")
print(df.isnull().sum())

print("\nDUPLICADOS")
print(df.duplicated().sum())

# ------------------------------------------------------------
# 4. LIMPIEZA DE NOMBRES DE COLUMNAS
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.upper()
    .str.replace(" ", "_")
)

print("\nCOLUMNAS NORMALIZADAS")
print(df.columns.tolist())

# ------------------------------------------------------------
# 5. ELIMINACIÓN DE COLUMNAS CON MUCHOS NULOS
# ------------------------------------------------------------

columnas_eliminar = [
    "AUTORIZA",
    "BONO",
    "PROMO",
    "REACREDITA",
    "COMPENSA",
    "USUARIO"
]

df = df.drop(columns=columnas_eliminar, errors="ignore")

print("\nCOLUMNAS DESPUÉS DE ELIMINAR NULOS")
print(df.columns.tolist())

# ------------------------------------------------------------
# 6. TRATAMIENTO DE NULOS RESTANTES
# ------------------------------------------------------------

# LOTE tiene pocos nulos, se rellena con 0
if "LOTE" in df.columns:
    df["LOTE"] = df["LOTE"].fillna(0)

# Variables numéricas
columnas_numericas = ["ACUMULA", "CONVALOR", "SINVALOR", "LOTE"]

for col in columnas_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

# Fechas
df["FECHA_TRAN"] = pd.to_datetime(df["FECHA_TRAN"], errors="coerce")
df["FECHA_PROC"] = pd.to_datetime(df["FECHA_PROC"], errors="coerce")

# Eliminar registros sin fecha de transacción
df = df.dropna(subset=["FECHA_TRAN"])

# ------------------------------------------------------------
# 7. CONSTRUCCIÓN DEL TARGET
# ------------------------------------------------------------

# TARGET = 1 si REDIME es REDE
# TARGET = 0 para el resto

df["TARGET"] = df["REDIME"].apply(lambda x: 1 if x == "REDE" else 0)

print("\nDISTRIBUCIÓN DEL TARGET")
print(df["TARGET"].value_counts())

# ------------------------------------------------------------
# 8. FEATURE ENGINEERING
# ------------------------------------------------------------

df["MES"] = df["FECHA_TRAN"].dt.month
df["DIA"] = df["FECHA_TRAN"].dt.day
df["DIA_SEMANA"] = df["FECHA_TRAN"].dt.dayofweek + 1

# Diferencia entre fecha de proceso y fecha de transacción
df["DIAS_PROCESO"] = (df["FECHA_PROC"] - df["FECHA_TRAN"]).dt.days
df["DIAS_PROCESO"] = df["DIAS_PROCESO"].fillna(0)

# Evitar valores negativos raros
df["DIAS_PROCESO"] = df["DIAS_PROCESO"].apply(lambda x: x if x >= 0 else 0)

# ------------------------------------------------------------
# 9. EDA ESTADÍSTICO
# ------------------------------------------------------------

total_registros = len(df)
clientes_unicos = df["CLIENTE"].nunique()
redenciones = df["TARGET"].sum()
no_redenciones = total_registros - redenciones
porcentaje_redencion = (redenciones / total_registros) * 100

print("\nRESUMEN GENERAL")
print("Total registros:", total_registros)
print("Clientes únicos:", clientes_unicos)
print("Redenciones:", redenciones)
print("No redenciones:", no_redenciones)
print("Porcentaje redención:", round(porcentaje_redencion, 4), "%")

print("\nDESCRIPCIÓN VARIABLES NUMÉRICAS")
print(df[["ACUMULA", "CONVALOR", "SINVALOR", "LOTE", "DIAS_PROCESO"]].describe())

print("\nDISTRIBUCIÓN REDIME")
print(df["REDIME"].value_counts())

print("\nTRANSACCIONES POR MES")
print(df["MES"].value_counts().sort_index())

print("\nTRANSACCIONES POR DÍA DE SEMANA")
print(df["DIA_SEMANA"].value_counts().sort_index())

# ------------------------------------------------------------
# 10. GRÁFICAS PARA EVIDENCIA
# ------------------------------------------------------------

# Gráfica 1: Distribución de REDIME
plt.figure(figsize=(8, 5))
df["REDIME"].value_counts().plot(kind="bar")
plt.title("Distribución de tipos de movimiento REDIME")
plt.xlabel("Tipo de movimiento")
plt.ylabel("Cantidad de registros")
plt.tight_layout()
plt.savefig("graficas/distribucion_redime.png")
plt.close()

# Gráfica 2: Distribución del TARGET
plt.figure(figsize=(6, 4))
df["TARGET"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribución del TARGET")
plt.xlabel("TARGET: 0 = No redime, 1 = Redime")
plt.ylabel("Cantidad de registros")
plt.tight_layout()
plt.savefig("graficas/distribucion_target.png")
plt.close()

# Gráfica 3: Puntos acumulados
plt.figure(figsize=(8, 5))
df["ACUMULA"].hist(bins=50)
plt.title("Distribución de puntos acumulados")
plt.xlabel("Puntos acumulados")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("graficas/distribucion_acumula.png")
plt.close()

# Gráfica 4: Transacciones por día de semana
plt.figure(figsize=(8, 5))
df["DIA_SEMANA"].value_counts().sort_index().plot(kind="bar")
plt.title("Transacciones por día de semana")
plt.xlabel("Día de semana")
plt.ylabel("Cantidad de transacciones")
plt.tight_layout()
plt.savefig("graficas/transacciones_dia_semana.png")
plt.close()

# ------------------------------------------------------------
# 11. GUARDAR DATASET LIMPIO
# ------------------------------------------------------------

df.to_csv(RUTA_CSV_LIMPIO, index=False, encoding="utf-8-sig")

print("\nDataset limpio guardado en:")
print(RUTA_CSV_LIMPIO)

# ------------------------------------------------------------
# 12. GUARDAR REPORTE DE EDA
# ------------------------------------------------------------

with open(RUTA_REPORTE_EDA, "w", encoding="utf-8") as reporte:
    reporte.write("REPORTE EDA - PREDICCIÓN DE REDENCIÓN DE PUNTOS\n")
    reporte.write("=================================================\n\n")
    reporte.write(f"Total de registros: {total_registros}\n")
    reporte.write(f"Clientes únicos: {clientes_unicos}\n")
    reporte.write(f"Redenciones: {redenciones}\n")
    reporte.write(f"No redenciones: {no_redenciones}\n")
    reporte.write(f"Porcentaje de redención: {round(porcentaje_redencion, 4)}%\n\n")

    reporte.write("Distribución REDIME:\n")
    reporte.write(str(df["REDIME"].value_counts()))
    reporte.write("\n\n")

    reporte.write("Valores nulos después de limpieza:\n")
    reporte.write(str(df.isnull().sum()))
    reporte.write("\n\n")

    reporte.write("Descripción estadística:\n")
    reporte.write(str(df[["ACUMULA", "CONVALOR", "SINVALOR", "LOTE", "DIAS_PROCESO"]].describe()))
    reporte.write("\n\n")

print("\nReporte EDA guardado en:")
print(RUTA_REPORTE_EDA)

print("\nGráficas generadas en carpeta graficas/")
print("EDA y limpieza finalizados correctamente.")
