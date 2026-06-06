from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, count, sum, avg, max, min, datediff, lit
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("Evaluacion_Real_Redencion_Cliente") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ============================================================
# 1. CARGA
# ============================================================

ruta_csv = "file:///root/transacciones_redencion_limpio.csv"

df = spark.read.csv(ruta_csv, header=True, inferSchema=True)

df = df.withColumn("FECHA_TRAN", col("FECHA_TRAN").cast("date"))
df = df.withColumn("ACUMULA", col("ACUMULA").cast(DoubleType()))
df = df.withColumn("LOTE", col("LOTE").cast(DoubleType()))
df = df.fillna(0, subset=["ACUMULA", "LOTE"])

print("\nDataset cargado")
print("Filas:", df.count())

print("\nFechas disponibles:")
df.select(
    min("FECHA_TRAN").alias("fecha_min"),
    max("FECHA_TRAN").alias("fecha_max")
).show()

print("\nDistribución REDIME:")
df.groupBy("REDIME").count().orderBy(col("count").desc()).show()

# ============================================================
# 2. CORTE TEMPORAL
# ============================================================

# Usamos historial antes del corte para predecir redenciones después.
# Ajusta la fecha si quieres probar otro escenario.
fecha_corte = "2026-02-15"

df_hist = df.filter(col("FECHA_TRAN") < lit(fecha_corte))
df_futuro = df.filter(col("FECHA_TRAN") >= lit(fecha_corte))

print("\nHistorial antes del corte:", df_hist.count())
print("Futuro después del corte:", df_futuro.count())

# ============================================================
# 3. FEATURES HISTÓRICAS POR CLIENTE
# ============================================================

features_cliente = df_hist.groupBy("CLIENTE").agg(
    count("*").alias("TOTAL_TRANSACCIONES_HIST"),
    sum("ACUMULA").alias("TOTAL_ACUMULA_HIST"),
    avg("ACUMULA").alias("PROMEDIO_ACUMULA_HIST"),
    max("ACUMULA").alias("MAX_ACUMULA_HIST"),
    min("ACUMULA").alias("MIN_ACUMULA_HIST"),
    avg("LOTE").alias("PROMEDIO_LOTE_HIST")
)

# ============================================================
# 4. TARGET FUTURO POR CLIENTE
# ============================================================

target_futuro = df_futuro.groupBy("CLIENTE").agg(
    sum(when(col("REDIME") == "REDE", 1).otherwise(0)).alias("REDENCIONES_FUTURAS")
)

target_futuro = target_futuro.withColumn(
    "TARGET",
    when(col("REDENCIONES_FUTURAS") > 0, 1).otherwise(0)
).select("CLIENTE", "TARGET")

# ============================================================
# 5. DATASET FINAL
# ============================================================

df_modelo = features_cliente.join(
    target_futuro,
    on="CLIENTE",
    how="left"
)

df_modelo = df_modelo.fillna(0, subset=["TARGET"])

df_modelo = df_modelo.withColumn("TARGET", col("TARGET").cast(IntegerType()))

print("\nDataset final por cliente")
print("Clientes:", df_modelo.count())

print("\nDistribución del TARGET futuro:")
df_modelo.groupBy("TARGET").count().show()

# ============================================================
# 6. VARIABLES
# ============================================================

features_cols = [
    "TOTAL_TRANSACCIONES_HIST",
    "TOTAL_ACUMULA_HIST",
    "PROMEDIO_ACUMULA_HIST",
    "MAX_ACUMULA_HIST",
    "MIN_ACUMULA_HIST",
    "PROMEDIO_LOTE_HIST"
]

for c in features_cols:
    df_modelo = df_modelo.withColumn(c, col(c).cast(DoubleType()))

df_modelo = df_modelo.fillna(0, subset=features_cols)

# ============================================================
# 7. PESO DE CLASES
# ============================================================

total = df_modelo.count()
positivos = df_modelo.filter(col("TARGET") == 1).count()
negativos = df_modelo.filter(col("TARGET") == 0).count()

peso_pos = total / (2.0 * positivos) if positivos > 0 else 1.0
peso_neg = total / (2.0 * negativos) if negativos > 0 else 1.0

df_modelo = df_modelo.withColumn(
    "classWeight",
    when(col("TARGET") == 1, peso_pos).otherwise(peso_neg)
)

print("\nPesos de clase")
print("Positivos:", positivos, "Peso:", peso_pos)
print("Negativos:", negativos, "Peso:", peso_neg)

# ============================================================
# 8. TRAIN / TEST
# ============================================================

train, test = df_modelo.randomSplit([0.8, 0.2], seed=42)

print("\nTrain:", train.count())
print("Test:", test.count())

print("\nDistribución en test:")
test.groupBy("TARGET").count().show()

# ============================================================
# 9. PIPELINE
# ============================================================

assembler = VectorAssembler(
    inputCols=features_cols,
    outputCol="features_raw",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False
)

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="TARGET",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

evaluator_pr = BinaryClassificationEvaluator(
    labelCol="TARGET",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="TARGET",
    predictionCol="prediction",
    metricName="accuracy"
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="TARGET",
    predictionCol="prediction",
    metricName="f1"
)

evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="TARGET",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="TARGET",
    predictionCol="prediction",
    metricName="weightedRecall"
)

resultados = []

def evaluar_modelo(nombre, modelo):
    print("\n============================================================")
    print("MODELO:", nombre)
    print("============================================================")

    pipeline = Pipeline(stages=[assembler, scaler, modelo])
    fitted = pipeline.fit(train)
    pred = fitted.transform(test)

    auc = evaluator_auc.evaluate(pred)
    pr_auc = evaluator_pr.evaluate(pred)
    accuracy = evaluator_accuracy.evaluate(pred)
    f1 = evaluator_f1.evaluate(pred)
    precision = evaluator_precision.evaluate(pred)
    recall = evaluator_recall.evaluate(pred)

    print("AUC-ROC:", round(auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Accuracy:", round(accuracy, 4))
    print("F1-Score:", round(f1, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))

    print("\nMatriz de confusión:")
    pred.groupBy("TARGET", "prediction").count().orderBy("TARGET", "prediction").show()

    print("\nEjemplo de predicciones:")
    pred.select(
        "CLIENTE",
        "TARGET",
        "prediction",
        "probability",
        "TOTAL_TRANSACCIONES_HIST",
        "TOTAL_ACUMULA_HIST",
        "PROMEDIO_ACUMULA_HIST"
    ).show(10, truncate=False)

    resultados.append({
        "Modelo": nombre,
        "AUC_ROC": round(auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "Accuracy": round(accuracy, 4),
        "F1_Score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    })

# ============================================================
# 10. MODELOS
# ============================================================

lr = LogisticRegression(
    featuresCol="features",
    labelCol="TARGET",
    weightCol="classWeight",
    maxIter=50,
    regParam=0.05
)

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="TARGET",
    weightCol="classWeight",
    maxDepth=4,
    minInstancesPerNode=20,
    seed=42
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="TARGET",
    weightCol="classWeight",
    numTrees=50,
    maxDepth=5,
    minInstancesPerNode=20,
    seed=42
)

evaluar_modelo("Logistic Regression", lr)
evaluar_modelo("Decision Tree", dt)
evaluar_modelo("Random Forest", rf)

# ============================================================
# 11. COMPARACIÓN FINAL
# ============================================================

print("\n============================================================")
print("COMPARACIÓN FINAL - EVALUACIÓN TEMPORAL")
print("============================================================")

resultados_df = spark.createDataFrame(resultados)
resultados_df.orderBy(col("PR_AUC").desc()).show(truncate=False)

salida = "file:///root/resultados_evaluacion_temporal"

resultados_df.coalesce(1).write.mode("overwrite").option("header", True).csv(salida)

print("\nResultados guardados en:")
print(salida)

print("""
CONCLUSIÓN:
Esta evaluación es más realista porque las variables predictoras se calculan
solo con historial anterior a la fecha de corte, mientras que el TARGET se
mide en el periodo futuro. Para datasets desbalanceados, PR-AUC es una métrica
más informativa que Accuracy, ya que se enfoca mejor en la clase minoritaria.
""")

spark.stop()