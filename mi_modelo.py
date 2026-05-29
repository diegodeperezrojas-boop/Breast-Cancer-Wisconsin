"""
╔══════════════════════════════════════════════════════╗
║        Mi Primer Modelo de Inteligencia Artificial   ║
║               Módulo III · Submódulo 2               ║
║  Alumno: Diego Peréz Rojas                           ║
╚══════════════════════════════════════════════════════╝

INSTRUCCIÓN: guarda este archivo y presiona F5 para ejecutar
"""

# ── LIBRERÍAS EXTERNAS ──────────────────────────────────────────────
import warnings # silencia avisos innecesarios
warnings.filterwarnings("ignore")

# matplotlib necesita esta línea ANTES de cualquier import de pyplot
# para que IDLE pueda abrir ventanas de gráficas
import matplotlib
matplotlib.use("TkAgg") # ← obligatorio para IDLE
import matplotlib.pyplot as plt
import numpy as np

# ── HERRAMIENTAS DE SCIKIT-LEARN ────────────────────────────────────
from sklearn.datasets import load_breast_cancer # el dataset
from sklearn.model_selection import train_test_split # dividir datos
from sklearn.ensemble import RandomForestClassifier # el modelo
from sklearn.metrics import (
    accuracy_score,       # % de aciertos totales
    precision_score,      # de lo que predije +, ¿cuánto era +?
    recall_score,         # de todos los + reales, ¿cuántos encontré?
    f1_score,             # media armónica de precision y recall
    confusion_matrix,     # tabla TP/TN/FP/FN
    classification_report # reporte completo
)

print("✅ Paso 1 completado — librerías cargadas correctamente")


# ── PASO 2: CARGAR EL DATASET ───────────────────────────────────────
# load_breast_cancer() descarga el dataset desde sklearn (viene incluido)
# No necesitas conexión a internet
data = load_breast_cancer() # objeto con todo el dataset

# Extraer X (características) e y (etiquetas)
X = data.data    # matriz de 569 × 30 — cada fila es un tumor
y = data.target  # vector de 569 — 0=maligno, 1=benigno

# ── EXPLORAR: ¿qué contiene el dataset? ────────────────────────────
print("=" * 60)
print("             DATASET: Breast Cancer Wisconsin")
print("=" * 60)
print(f" Total de muestras       : {X.shape[0]}")
print(f" Características/muestra : {X.shape[1]}")
print(f" Nombres de clases       : {list(data.target_names)}")
print()
print(f" Tumores MALIGNOS        : {sum(y==0)} (clase 0)")
print(f" Tumores BENIGNOS        : {sum(y==1)} (clase 1)")
print()

# Ver los nombres de las 30 características
print(" Las 30 características medidas en cada tumor:")
for i, nombre in enumerate(data.feature_names):
    print(f"   {i+1:>2}. {nombre}")
print()

# Ver los valores de la primera muestra (primer tumor)
print(" Primera muestra (tumor #1):")
print(f"   Primeras 5 mediciones: {X[0, :5].round(2)}")
print(f"   Etiqueta real: {data.target_names[y[0]]}")

print("\n✅ Paso 2 completado — dataset cargado")


# ── PASO 3: DIVIDIR EL DATASET ──────────────────────────────────────
# train_test_split() mezcla y divide los datos automáticamente
#
# PARÁMETRO IMPORTANTE: test_size
# → controla qué porcentaje va a prueba
# → 0.20 = 20% prueba, 80% entrenamiento
# → valores típicos: 0.15, 0.20, 0.25, 0.30
#
# PARÁMETRO: random_state
# → "semilla" de aleatoriedad
# → si usas el mismo número, siempre obtienes la misma división
# → útil para que dos personas obtengan los mismos resultados

X_train, X_test, y_train, y_test = train_test_split(
    X,                # datos de entrada (features)
    y,                # etiquetas (maligno/benigno)
    test_size = 0.20, # ← hiperparámetro: % de datos para prueba
    random_state = 42 # ← semilla de aleatoriedad (cualquier número)
)

print("=" * 60)
print("                     DIVISIÓN DEL DATASET")
print("=" * 60)
print(f" Total de muestras         : {len(X)}")
print(f" Entrenamiento (80%)       : {len(X_train)} muestras")
print(f" Prueba (20%)              : {len(X_test)} muestras")
print()
print(f" Malignos en entrenamiento : {sum(y_train==0)}")
print(f" Benignos en entrenamiento : {sum(y_train==1)}")
print(f" Malignos en prueba        : {sum(y_test==0)}")
print(f" Benignos en prueba        : {sum(y_test==1)}")

print("\n✅ Paso 3 completado — datos divididos")

# ── PASO 4: MODELO ───────────────────────────────

from sklearn.ensemble import RandomForestClassifier

N_ESTIMATORS = 10
MAX_DEPTH = 3
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1

modelo = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=42
)

print("=" * 60)
print("MODELO CREADO — Random Forest Classifier")
print("=" * 60)

print(f"n_estimators : {N_ESTIMATORS} (árboles)")
print(f"max_depth : {MAX_DEPTH}")
print(f"min_samples_split : {MIN_SAMPLES_SPLIT}")
print(f"min_samples_leaf : {MIN_SAMPLES_LEAF}")

print("\n⚙ Modelo creado pero NO entrenado aún.")
print("➡ El entrenamiento será en el Paso 5")

print("\n Paso 4 completado")

#── PASO 5: ENTRENAR EL MODELO ──────────────────────────────────────

# .fit(X_train, y_train) = el modelo estudia los datos de entrenamiento
#
# Lo que ocurre internamente:
# 1. Construye N_ESTIMATORS árboles de decisión
# 2. Cada árbol usa una muestra aleatoria de los datos (bagging)
# 3. En cada nodo, elige la mejor característica para dividir
# 4. Los árboles quedan guardados en memoria como PARÁMETROS del modelo

print(" Entrenando el modelo... ", end="", flush=True)

modelo.fit(X_train, y_train)   # ← TODO el entrenamiento en una línea

print("listo ✓")
print()

# Curiosidad: ¿cuántos árboles se crearon realmente?
n_arboles_reales = len(modelo.estimators_)

print(f" Árboles creados : {n_arboles_reales}")
print(f" Clases aprendidas : {list(modelo.classes_)} → 0=maligno, 1=benigno")
print(f" Características usadas : {modelo.n_features_in_}")

# El árbol #0 (el primero) — ver su profundidad real
print(f" Profundidad árbol #1 : {modelo.estimators_[0].get_depth()}")

print("\n✅ Paso 5 completado — modelo entrenado")
print(" Ahora el modelo conoce los patrones de 455 biopsias.")

# ── PASO 6: PREDECIR Y CALCULAR MÉTRICAS ────────────────────────────

# .predict() usa los árboles entrenados para predecir las 114 muestras
y_pred = modelo.predict(X_test)   # array con 114 predicciones: 0 ó 1


# ── CALCULAR LAS 4 MÉTRICAS ─────────────────────────────────────────

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)


# ── MOSTRAR RESULTADOS ──────────────────────────────────────────────

print("=" * 60)
print(f" RESULTADOS — n_est={N_ESTIMATORS}, depth={MAX_DEPTH}")
print("=" * 60)

print(f" Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f" Precision : {precision:.4f}")
print(f" Recall    : {recall:.4f} ← prioritario (tumores malignos)")
print(f" F1-Score  : {f1:.4f}")

print()


# Reporte completo por clase
print(" Reporte detallado:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# ── CALCULAR LA MATRIZ Y VER LOS FALSOS NEGATIVOS ───────────────────

cm_vals = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm_vals.ravel()   # desempaquetar los 4 valores


print(" Desglose de predicciones:")

print(f" TP = {TP:>3} → benignos correctamente detectados")
print(f" TN = {TN:>3} → malignos correctamente detectados")

print(f" FP = {FP:>3} → malignos clasificados como benignos (riesgo)")
print(f" FN = {FN:>3} → benignos clasificados como malignos (molestia)")

print()

print(f" ⚠ {FP} tumores malignos NO detectados (FP clase maligno)")

print("\n✅ Paso 6 completado — métricas calculadas")
# ── PASO 7: VISUALIZAR LA MATRIZ DE CONFUSIÓN ───────────────────────

# Creamos la gráfica con matplotlib puro (sin seaborn)
# para máxima compatibilidad con IDLE

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.suptitle(
    f"Matriz de Confusión — n_est={N_ESTIMATORS}, depth={MAX_DEPTH}",
    fontsize=13,
    fontweight="bold"
)


# ── Panel izquierdo: valores absolutos ──────────────────────────────

valores = np.array([
    [TN, FP],
    [FN, TP]
], dtype=float)

etiquetas_rc = ["Maligno (real)", "Benigno (real)"]
etiquetas_cc = ["Pred: Maligno", "Pred: Benigno"]

nombres_celdas = [
    ["TN", "FP"],
    ["FN ⚠", "TP"]
]

ax1 = axes[0]

img1 = ax1.imshow(valores, cmap="Blues")


for i in range(2):
    for j in range(2):

        val = int(valores[i, j])

        color = (
            "white"
            if valores[i, j] > valores.max() * 0.5
            else "black"
        )

        ax1.text(
            j,
            i,
            f"{nombres_celdas[i][j]}\n{val}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=color
        )


ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])

ax1.set_xticklabels(etiquetas_cc, fontsize=10)
ax1.set_yticklabels(etiquetas_rc, fontsize=10)

ax1.set_xlabel("Predicción del modelo", fontsize=11)
ax1.set_ylabel("Valor real", fontsize=11)

ax1.set_title("Valores absolutos", fontsize=11)

plt.colorbar(img1, ax=ax1)


# ── Panel derecho: barras de métricas ───────────────────────────────

ax2 = axes[1]

metr_nombres = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

metr_valores = [
    accuracy,
    precision,
    recall,
    f1
]

colores_barras = [
    "#378ADD",
    "#f97316",
    "#ef4444",
    "#7c3aed"
]

barras = ax2.bar(
    metr_nombres,
    metr_valores,
    color=colores_barras,
    alpha=0.85,
    edgecolor="white"
)


for barra, val in zip(barras, metr_valores):

    ax2.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 0.005,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )


ax2.set_ylim(0.0, 1.08)

ax2.set_title("Métricas de rendimiento", fontsize=11)
ax2.set_ylabel("Valor", fontsize=11)

ax2.axhline(
    0.9,
    color="gray",
    linestyle="--",
    linewidth=1,
    label="umbral 90%"
)

ax2.legend(fontsize=9)

ax2.grid(axis="y", alpha=0.3)


# ── AJUSTAR Y GUARDAR ───────────────────────────────────────────────

plt.tight_layout()


# Guardar la imagen como archivo PNG
nombre_img = f"resultado_n{N_ESTIMATORS}_d{MAX_DEPTH}.png"

plt.savefig(
    nombre_img,
    dpi=120,
    bbox_inches="tight"
)

print(f"\n Imagen guardada → {nombre_img}")


# ← abre la ventana gráfica en IDLE
plt.show()

print("\n✅ Paso 7 completado — gráfica generada")

# ── PASO 8: COMPARAR CONFIGURACIONES ────────────────────────────────

# Entrenamos el modelo 5 veces con distintos hiperparámetros
# y comparamos los resultados en una tabla


# ════════════════════════════════════════════════════════════
# ✏ CONFIGURA TUS 5 COMBINACIONES AQUÍ
# ════════════════════════════════════════════════════════════

configuraciones = [

    {
        "nombre": "Config A — base",
        "n_estimators": 10,
        "max_depth": 3,
        "min_samples_leaf": 1
    },

    {
        "nombre": "Config B — +árboles",
        "n_estimators": 50,
        "max_depth": 3,
        "min_samples_leaf": 1
    },

    {
        "nombre": "Config C — +profund.",
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_leaf": 1
    },

    {
        "nombre": "Config D — +regular.",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 3
    },

    {
        "nombre": "Config E — sin límite",
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    },
]


# ════════════════════════════════════════════════════════════

tabla = []

print(" Entrenando 5 configuraciones", end="", flush=True)

for cfg in configuraciones:

    m = RandomForestClassifier(
        n_estimators   = cfg["n_estimators"],
        max_depth      = cfg["max_depth"],
        min_samples_leaf = cfg["min_samples_leaf"],
        random_state   = 42
    )

    m.fit(X_train, y_train)

    yp = m.predict(X_test)

    c = confusion_matrix(y_test, yp)

    tn2, fp2, fn2, tp2 = c.ravel()

    tabla.append({

        "nombre"    : cfg["nombre"],
        "n_est"     : cfg["n_estimators"],
        "depth"     : str(cfg["max_depth"]),
        "leaf"      : cfg["min_samples_leaf"],

        "TP"        : int(tp2),
        "FP"        : int(fp2),
        "FN"        : int(fn2),

        "Accuracy"  : round(accuracy_score(y_test, yp), 4),
        "Precision" : round(precision_score(y_test, yp), 4),
        "Recall"    : round(recall_score(y_test, yp), 4),
        "F1"        : round(f1_score(y_test, yp), 4),
    })

    print(".", end="", flush=True)

print(" listo ✓\n")


# ── IMPRIMIR TABLA ──────────────────────────────────────────────────

SEP = "=" * 105

print(SEP)
print(" TABLA COMPARATIVA DE HIPERPARÁMETROS")
print(SEP)

print(
    f" {'Configuración':<22} {'n_est':>6} {'depth':>7} {'leaf':>5}"
    f" {'TP':>4} {'FP':>4} {'FN':>4}"
    f" {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
)

print("-" * 105)

mejor_f1 = max(r["F1"] for r in tabla)
menor_fn = min(r["FN"] for r in tabla)


for r in tabla:

    marca = (
        " ← mejor F1"
        if r["F1"] == mejor_f1
        else (
            " ← menos FN"
            if r["FN"] == menor_fn
            else ""
        )
    )

    print(
        f" {r['nombre']:<22}"
        f" {r['n_est']:>6}"
        f" {r['depth']:>7}"
        f" {r['leaf']:>5}"
        f" {r['TP']:>4}"
        f" {r['FP']:>4}"
        f" {r['FN']:>4}"
        f" {r['Accuracy']:>9.4f}"
        f" {r['Precision']:>10.4f}"
        f" {r['Recall']:>8.4f}"
        f" {r['F1']:>8.4f}"
        f"{marca}"
    )

print(SEP)


# ── GRÁFICA COMPARATIVA ─────────────────────────────────────────────

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

fig2.suptitle(
    "Comparación de 5 configuraciones de hiperparámetros",
    fontsize=13,
    fontweight="bold"
)


# ── Panel izquierdo: métricas ───────────────────────────────────────

nombres_conf = [r["nombre"].strip() for r in tabla]

metricas_comp = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

col_comp = [
    "#378ADD",
    "#f97316",
    "#ef4444",
    "#7c3aed"
]

xpos = np.arange(len(tabla))

ancho = 0.18


for i, (met, col) in enumerate(zip(metricas_comp, col_comp)):

    vals = [r[met] for r in tabla]

    ax3.bar(
        xpos + i * ancho,
        vals,
        ancho,
        label=met,
        color=col,
        alpha=0.85,
        edgecolor="white"
    )


ax3.set_xticks(xpos + 1.5 * ancho)

ax3.set_xticklabels(
    nombres_conf,
    rotation=20,
    ha="right",
    fontsize=9
)

ax3.set_ylim(0.80, 1.03)

ax3.set_title("Métricas por configuración")

ax3.legend(fontsize=9)

ax3.grid(axis="y", alpha=0.3)


# ── Panel derecho: falsos negativos ─────────────────────────────────

fns_comp = [r["FN"] for r in tabla]

col_fn = [

    "#22c55e" if v == min(fns_comp)
    else "#ef4444" if v == max(fns_comp)
    else "#f97316"

    for v in fns_comp
]


bars_fn = ax4.bar(
    nombres_conf,
    fns_comp,
    color=col_fn,
    alpha=0.88,
    edgecolor="white"
)


for b, v in zip(bars_fn, fns_comp):

    ax4.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.1,
        str(v),
        ha="center",
        fontsize=12,
        fontweight="bold"
    )


ax4.set_xticklabels(
    nombres_conf,
    rotation=20,
    ha="right",
    fontsize=9
)

ax4.set_title("Falsos Negativos por configuración")

ax4.set_ylabel("Cantidad de FN")

ax4.grid(axis="y", alpha=0.3)


# ── AJUSTAR Y GUARDAR ───────────────────────────────────────────────

plt.tight_layout()

plt.savefig(
    "comparacion_5configs.png",
    dpi=120,
    bbox_inches="tight"
)

print(" Gráfica guardada → comparacion_5configs.png")


# Mostrar ventana gráfica
plt.show()


# ── FINAL ───────────────────────────────────────────────────────────

print("\n✅ Paso 8 completado — comparación lista")

print(f" Mejor F1-Score: {mejor_f1}")

print(f" Config con menos FN: ver tabla arriba")


# Línea final para que IDLE no cierre la ventana
# antes de que veas los resultados

print("\n Programa completado. Presiona Enter para salir.")

input()
