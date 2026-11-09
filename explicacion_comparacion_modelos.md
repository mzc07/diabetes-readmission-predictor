# Explicación completa de las diapositivas de comparación de modelos

Proyecto: predicción de reingreso hospitalario en **<30 días** en pacientes diabéticos (dataset *Diabetes 130-US hospitals*).
Diapositivas cubiertas: **6 a 12** (comparación con gráficas, métrica ganadora, selección del modelo y justificación).

Todos los gráficos los genera `src/evaluation.py` y se invocan desde `train.py`. Las figuras están en `reports/figures/` y las tablas en `artifacts/`.

---

## 0. Contexto necesario antes de leer los gráficos

### 0.1 Los datos y el desbalance

| Dato | Valor |
|---|---|
| Pacientes (1 encuentro por paciente) | 69,990 |
| Positivos (reingresan <30 días) | 9.0 % |
| Split | 80 % train / 20 % test, estratificado (`random_state=42`) |
| Test | 13,998 pacientes = **1,257 positivos** y **12,741 negativos** |

El valor 9 % es clave: es la **línea base** contra la cual se compara todo. Un modelo que marque a todos como "reingresa" tendría precision ≈ 0.09 y recall = 1.0 (F1 ≈ 0.165). Un modelo que no marque a nadie tendría ~91 % de accuracy pero recall 0.

### 0.2 Los 4 modelos

| Modelo | Configuración (en `src/models.py`) |
|---|---|
| GaussianNB | Sin `class_weight` (no lo soporta) |
| RandomForest | 200 árboles, `min_samples_leaf=20`, `class_weight='balanced'` |
| DecisionTree | `max_depth=5`, `min_samples_leaf=50`, `class_weight='balanced'` |
| SVM | Kernel RBF, `class_weight='balanced'` |

Todos usan el mismo pipeline: **escalar/one-hot → seleccionar top 20 columnas → modelo**, todo ajustado solo con train.

### 0.3 Flujo de evaluación (importante para entender de dónde sale cada gráfico)

1. Del conjunto de train se toma una **submuestra de 20,000 filas** (por velocidad, sobre todo del SVM).
2. Sobre esa submuestra se hace **validación cruzada estratificada de 5 folds** → alimenta los gráficos de la diapositiva 6.
3. Los 4 modelos se reentrenan con la submuestra completa y se evalúan en el **test (20 %)** → alimenta las diapositivas 7, 8 y 9.
4. El ganador se elige por **F1 medio en CV** (no por test) y se reentrena con **todo** el train → `artifacts/winner.json` (diapositiva 11).

### 0.4 Definiciones de las métricas

Se usa la clase positiva = "reingresa <30 días". De la matriz de confusión:

- **TP** (verdadero positivo): predijo reingreso y sí reingresó.
- **FP** (falso positivo): predijo reingreso pero no reingresó.
- **FN** (falso negativo): predijo no reingreso pero sí reingresó.
- **TN** (verdadero negativo): predijo no reingreso y no reingresó.

| Métrica | Fórmula | Pregunta que responde |
|---|---|---|
| **Precision** | TP / (TP + FP) | De los que el modelo marca en riesgo, ¿qué fracción realmente reingresa? |
| **Recall** (sensibilidad) | TP / (TP + FN) | De los que realmente reingresan, ¿qué fracción detecta el modelo? |
| **F1** | 2·P·R / (P + R) | Media armónica de precision y recall. Penaliza si cualquiera de los dos es bajo. |
| **AUC (ROC)** | Área bajo la curva ROC | Probabilidad de que un paciente que sí reingresa reciba un puntaje mayor que uno que no reingresa. 0.5 = azar, 1.0 = perfecto. **No depende del umbral.** |

En el código (`compute_metrics`): `f1_score`, `precision_score`, `recall_score` usan `model.predict` (umbral por defecto: 0.5 en probabilidad / 0 en `decision_function`); `roc_auc_score` usa el puntaje continuo de `risk_score`.

### 0.5 El "puntaje de riesgo" y el umbral

Los modelos producen un puntaje continuo y luego se decide un **umbral**: si puntaje ≥ umbral → "en riesgo".

- GaussianNB, RandomForest, DecisionTree: `predict_proba(X)[:, 1]`.
- SVM: no se entrenó con `probability=True` (sería muy lento), así que se usa `decision_function` pasada por una sigmoide `1/(1+e^-x)` para dejarlo entre 0 y 1. **Ojo:** ese número ordena bien a los pacientes, pero **no es una probabilidad calibrada**.

---

## 1. Diapositiva 6 — Validación cruzada (boxplot y líneas por fold)

Archivos: `cv_boxplot.png` y `cv_lineas.png`. Datos: `artifacts/cv_results.csv`.

### 1.1 Cómo se calculó

`cross_validate_models` usa `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre la submuestra de 20,000 filas. Para cada modelo:

1. Divide los datos en 5 partes (folds) con la misma proporción de positivos (~9 %).
2. Entrena con 4 partes y mide **F1 de la clase <30** en la parte restante.
3. Repite 5 veces → **5 valores de F1 por modelo**.

El preprocesamiento y la selección de columnas están *dentro* del pipeline, por lo que se reajustan en cada fold: no hay fuga de información.

Valores obtenidos:

| Modelo | F1 por fold | Media | Desv. |
|---|---|---|---|
| GaussianNB | 0.199, 0.177, 0.191, 0.210, 0.217 | **0.199** | ±0.016 |
| RandomForest | 0.230, 0.223, 0.216, 0.230, 0.218 | **0.224** | ±0.006 |
| DecisionTree | 0.208, 0.216, 0.205, 0.215, 0.207 | **0.210** | ±0.005 |
| SVM | 0.214, 0.215, 0.214, 0.224, 0.203 | **0.214** | ±0.007 |

### 1.2 Gráfico de cajas (boxplot)

**Qué representa:** la distribución de los 5 F1 de cada modelo.

- **Línea central de la caja:** la mediana.
- **Caja:** del percentil 25 al 75 (rango intercuartílico).
- **Bigotes:** valores dentro de 1.5 × el rango intercuartílico.
- **Círculos sueltos:** valores atípicos. En SVM, los folds 4 (0.224) y 5 (0.203) aparecen como círculos porque los otros 3 folds son casi idénticos (≈0.214) y la caja queda colapsada en una línea.

**Cómo interpretarlo:**
- *Altura de la caja* → qué tan bueno es el modelo (más arriba = mejor F1).
- *Tamaño de la caja* → qué tan estable es. GaussianNB tiene la caja más ancha: su resultado depende mucho de qué datos le toquen.
- RandomForest queda más arriba que todos, pero su caja (0.217–0.230) **se solapa** con la de SVM y DecisionTree: la ventaja es real en promedio pero pequeña.

### 1.3 Gráfico de líneas por fold

**Qué representa:** el mismo dato desagregado: eje X = número de fold, eje Y = F1, una línea por modelo.

**Cómo interpretarlo:**
- Permite ver si un modelo gana **de forma consistente** (en todos los folds) o solo por un fold con suerte.
- Como los 4 modelos se evalúan sobre **los mismos folds**, una caída simultánea de todas las líneas indica un fold "difícil", no un fallo de un modelo en particular.
- Las líneas se cruzan varias veces: otra señal de que las diferencias son pequeñas.

### 1.4 Conclusión de la diapositiva

RandomForest tiene el F1 medio más alto (0.224) y poca variación. Es la base para elegirlo ganador, pero con una diferencia de ~0.01–0.025 sobre los demás.

---

## 2. Diapositiva 7 — Resultados en test (barras de métricas y matrices de confusión)

Archivos: `comparativa_metricas.png`, `matrices_confusion.png`. Datos: `artifacts/metrics.csv`.

### 2.1 Gráfico de barras de métricas

**Cómo se calculó:** `compute_metrics` evalúa cada modelo (entrenado con la submuestra de 20k) sobre el **test completo** (13,998 pacientes que ninguno vio). Las 4 métricas se pasan a formato largo y se grafican agrupadas por métrica, una barra por modelo.

| Modelo | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| GaussianNB | 0.2095 | **0.2010** | 0.2188 | **0.6407** |
| RandomForest | **0.2252** | 0.1457 | 0.4956 | 0.6399 |
| DecisionTree | 0.2059 | 0.1234 | **0.6205** | 0.6237 |
| SVM | 0.2104 | 0.1345 | 0.4837 | 0.6185 |

**Cómo interpretarlo:**
- **F1:** las barras son casi iguales (0.206–0.225). RandomForest gana por poco.
- **Precision:** todas por debajo de 0.20. Línea base = 0.09, así que el mejor modelo apenas duplica el azar.
- **Recall:** aquí sí hay diferencias grandes. DecisionTree detecta 62 % de los reingresos y GaussianNB solo 22 %.
- **AUC:** todas entre 0.62 y 0.64.
- Hay un **compromiso (trade-off)** visible: GaussianNB tiene la mayor precision pero el menor recall; DecisionTree, al revés. El F1 balancea ambos y por eso RandomForest gana.

### 2.2 Matrices de confusión

**Qué representa:** para cada modelo, una tabla 2×2 con el conteo de pacientes del test. Filas = realidad (True label), columnas = predicción (Predicted label). Se generan con `ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test))`, es decir, **con el umbral por defecto**.

| Modelo | TN | FP | FN | TP |
|---|---|---|---|---|
| GaussianNB | 11,648 | 1,093 | 982 | 275 |
| RandomForest | 9,089 | 3,652 | 634 | 623 |
| DecisionTree | 7,201 | 5,540 | 477 | 780 |
| SVM | 8,827 | 3,914 | 649 | 608 |

Cada fila suma el total real: 12,741 negativos y 1,257 positivos.

**Cómo leerla** (ejemplo RandomForest):
- Arriba-izquierda (9,089): no reingresaban y se predijo bien.
- Arriba-derecha (3,652): **falsas alarmas**.
- Abajo-izquierda (634): reingresos **que se escaparon**.
- Abajo-derecha (623): reingresos detectados.
- Precision = 623 / (623 + 3,652) = **0.146**. Recall = 623 / (623 + 634) = **0.496**.

**Qué enseña:**
- El color más oscuro (más pacientes) está en la celda de TN en todos. El problema es que la celda TP es pequeña.
- Para detectar la mitad de los reingresos, RandomForest genera ~3,650 falsas alarmas: por cada acierto hay unas 6 falsas alarmas.
- GaussianNB parece el "mejor" si solo se mira *accuracy* (85 %, porque casi no predice positivos), pero detecta apenas 275 de 1,257 reingresos. Este es exactamente el motivo por el que **no se usa accuracy**.

Accuracy calculada (solo como ilustración; no se usa para decidir): GaussianNB 85.2 %, RandomForest 69.4 %, SVM 67.4 %, DecisionTree 57.0 %. Predecir "nadie reingresa" daría 91.0 %.

---

## 3. Diapositiva 8 — Umbral de decisión (un gráfico por modelo)

Archivos: `umbral_GaussianNB.png`, `umbral_RandomForest.png`, `umbral_DecisionTree.png`, `umbral_SVM.png`.

### 3.1 Cómo se calculó

`plot_threshold_curves`, para cada modelo:

1. Calcula el puntaje de riesgo de cada paciente del test.
2. `precision_recall_curve(y_test, scores)` devuelve precision y recall para **cada umbral posible**.
3. F1 = 2·P·R / (P + R) para cada umbral.
4. Marca con una línea vertical gris punteada el umbral donde el F1 es máximo.

### 3.2 Qué representa

- Eje X: umbral (a partir de qué puntaje se marca "en riesgo").
- Eje Y: valor de la métrica (0 a 1).
- Tres curvas: **precision** (azul), **recall** (naranja), **F1** (verde).

### 3.3 Cómo interpretarlo

- **Umbral bajo** (izquierda): casi todos son marcados en riesgo → recall ≈ 1.0, pero precision ≈ 0.09 (la prevalencia). F1 ≈ 0.165 (esa meseta es el F1 de "marcar a todos").
- **Umbral alto** (derecha): solo se marcan los casos más seguros → precision sube, recall se desploma (y F1 también). La precision se vuelve ruidosa al final porque quedan muy pocos pacientes marcados.
- El **pico del F1** es el mejor equilibrio. Se alcanza en un valor estrecho y bajo: ~0.23–0.24 en RandomForest (umbral 0.54), ~0.22 en el resto.

| Modelo | Mejor umbral | Forma de la curva y por qué |
|---|---|---|
| RandomForest | 0.54 | Curvas suaves; pico claro de F1 (~0.235). |
| DecisionTree | 0.55 | **Curvas escalonadas**: con profundidad 5 hay solo unas pocas decenas de valores de probabilidad distintos. |
| SVM | 0.56 | Suave. La precision se dispara al final con muy pocos casos (ruido). El "umbral" es sobre la sigmoide, no una probabilidad real. |
| GaussianNB | 0.03 | **F1 casi plano** en toda la gráfica. Naive Bayes da probabilidades extremas (casi 0 o casi 1) porque asume independencia entre variables; por eso el mejor umbral es 0.03 y entre 0.1 y 0.9 apenas cambia nada. |

### 3.4 Qué enseña

- El umbral por defecto (0.5) **no es especial**. Se puede elegir un umbral según el costo clínico: priorizar recall (no perder reingresos) a costa de más falsas alarmas, o precision (menos falsas alarmas) a costa de perder casos.
- La curva de F1 es plana alrededor de su máximo: cambiar el umbral un poco casi no mejora el F1. No hay un "punto mágico" que resuelva el problema.
- **Advertencia metodológica:** este "mejor umbral" se calcula sobre el test, así que es una referencia ilustrativa, no una elección limpia. Para fijar un umbral real habría que hacerlo con validación y luego confirmar en test.

---

## 4. Diapositiva 9 — Curvas comparativas (F1 vs. umbral, PR y ROC)

Archivos: `comparativa_umbral_f1.png`, `comparativa_pr.png`, `comparativa_roc.png`. Son los mismos puntajes del test, ahora con los 4 modelos superpuestos (`plot_comparison`).

### 4.1 F1 vs. umbral (todos los modelos)

**Qué es:** las curvas verdes de la sección 3 superpuestas.

**Cómo interpretarla:**
- Todas parten del mismo valor (~0.165, el F1 de marcar a todos) y terminan en 0.
- RandomForest alcanza el pico más alto (~0.235 hacia umbral 0.54), seguido de DecisionTree y SVM (~0.22).
- GaussianNB es casi horizontal en ~0.21–0.22 en casi todo el rango.
- Los picos difieren en ~0.01–0.02. Es la forma visual de decir "ventaja modesta".
- Los umbrales óptimos de cada modelo **no son comparables entre sí** (por la diferente naturaleza de sus puntajes), por eso un mismo umbral no sirve para comparar modelos de manera justa.

### 4.2 Curva Precision-Recall (PR)

**Cómo se calculó:** `precision_recall_curve` barre todos los umbrales; cada punto es un par (recall, precision). Eje X = recall, eje Y = precision.

**Cómo interpretarla:**
- Una curva **más arriba y a la derecha** es mejor.
- La línea base de un clasificador aleatorio es una horizontal en y = 0.09 (prevalencia). **Las curvas terminan ahí** (recall = 1 → precision = 0.09).
- A recall bajo (< 0.1) hay picos de precision (0.3–0.6) producidos por muy pocos pacientes: no son fiables.
- Entre recall 0.1 y 0.4 las curvas se mueven en precision 0.15–0.30 y **se superponen**. RandomForest y DecisionTree quedan ligeramente por encima; SVM un poco por debajo.
- Para recall ≥ 0.5, **las 4 curvas son prácticamente la misma línea** (precision ≈ 0.10–0.14).
- Esta curva es más informativa que la ROC cuando hay clases desbalanceadas, porque se centra en la clase minoritaria.

### 4.3 Curva ROC

**Cómo se calculó:** `roc_curve` barre los umbrales; cada punto es (FPR, TPR), donde
- FPR = FP / (FP + TN) (tasa de falsos positivos, eje X)
- TPR = recall = TP / (TP + FN) (eje Y)

La leyenda muestra el AUC de cada modelo (`roc_auc_score`). La diagonal punteada gris es el azar (AUC = 0.5).

**Cómo interpretarla:**
- Más cerca de la esquina superior izquierda = mejor. Una curva sobre la diagonal indica que el modelo **ordena mejor que el azar**.
- AUC = probabilidad de que un paciente que sí reingresa tenga mayor puntaje que uno que no.
- Aquí todas las curvas quedan **muy cerca de la diagonal** y entre sí: AUC de 0.62 a 0.64.
- Escala orientativa: 0.5 azar · 0.6–0.7 pobre · 0.7–0.8 aceptable · 0.8–0.9 bueno · >0.9 excelente. Estos modelos están en la franja "pobre".

### 4.4 Conclusión de la diapositiva

> "Las curvas de los 4 modelos están muy juntas… ningún modelo domina claramente al resto."

Cuando las curvas se superponen, el modelo no se está distinguiendo por su algoritmo sino por el **límite de información que contienen los datos**. Esto conecta con la sección 7.

---

## 5. Diapositiva 10 — Por qué F1 como métrica ganadora

- **Accuracy engaña:** con 9 % de positivos, "nadie reingresa" daría ~91 % de accuracy y detectaría cero reingresos.
- **F1 de la clase <30:** mide solo la clase que importa y exige equilibrio entre precision y recall.
- **Selección por CV, no por test:** el ganador se elige con el F1 medio de 5 folds. El test queda como evaluación final independiente.
- **Precision, recall y AUC** son métricas de apoyo, para entender cómo se logra el F1.

---

## 6. Diapositivas 11 y 12 — Modelo ganador

### 6.1 Cifras

| Origen | Entrenamiento | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|
| CV (5 folds) | submuestra 20k | 0.224 ± 0.006 | – | – | – |
| Test (`metrics.csv`) | submuestra 20k | 0.2252 | 0.1457 | 0.4956 | 0.6399 |
| Test (`winner.json`, **diapositiva 11**) | **todo el train** | 0.2236 | 0.1454 | 0.4837 | 0.6478 |

La diapositiva 11 muestra el modelo final (reentrenado con todo el train); la diapositiva 12 cita el 0.225 del modelo entrenado con la submuestra. Son dos corridas distintas del mismo modelo, por eso difieren en milésimas. Con 5× más datos de entrenamiento el F1 casi no cambió (0.2252 → 0.2236) y el AUC subió solo 0.008: **más datos no es la solución**.

### 6.2 Por qué RandomForest (resumen de la diapositiva 12)

1. Mayor F1 medio en CV y en test.
2. Más estable que GaussianNB (±0.006 frente a ±0.016).
3. Recall ≈ 0.5 (frente a 0.22 de GaussianNB).
4. Captura relaciones no lineales e interacciones.
5. `class_weight='balanced'` compensa el desbalance y entrega importancias de columnas.
6. **Limitación honesta:** la ventaja sobre los otros es modesta.

---

## 7. ¿Por qué las métricas son tan bajas?

F1 ≈ 0.22, precision ≈ 0.15 y AUC ≈ 0.64 parecen malos, pero **no indican un error de código**: el pipeline es correcto (sin fuga de datos, validación cruzada, split estratificado). Son el resultado esperable en este problema. Las causas, de mayor a menor peso:

### 7.1 La información disponible no contiene suficiente señal (causa principal)

- Los datos son registros administrativos de una hospitalización: diagnóstico principal, número de visitas, medicamentos, etc. **No incluyen** lo que realmente determina un reingreso a 30 días: severidad clínica real, valores de laboratorio (solo hay "A1C/glucosa medida o no"), adherencia al tratamiento, calidad del seguimiento tras el alta, apoyo familiar, nivel socioeconómico, acceso a atención ambulatoria.
- Que **cuatro algoritmos muy distintos** (probabilístico, árboles, bosque y kernel) lleguen al mismo AUC (0.62–0.65) es la señal más clara: el límite lo pone la información de entrada, no el algoritmo.
- La literatura sobre este mismo dataset reporta AUC de alrededor de 0.60–0.70 con modelos mucho más elaborados, así que el resultado es coherente con lo conocido.

### 7.2 La clase positiva es rara (9 %)

- Con pocos positivos, incluso un modelo decente produce muchas falsas alarmas. Como hay ~10 negativos por cada positivo, para que la precision fuera de 0.5 el modelo tendría que discriminar con muchísima finura.
- Precision máxima útil: la línea base es 0.09. Una precision de 0.145 significa **1.6 veces** mejor que elegir al azar (lift = 0.1457 / 0.0898).
- Con precision tan baja, el F1 queda acotado: F1 = 2·0.145·0.496 / (0.145 + 0.496) ≈ 0.225.

### 7.3 `class_weight='balanced'` mueve el punto de operación

Para compensar el desbalance, tres modelos penalizan más los errores sobre positivos. Eso **sube el recall** (0.48–0.62) pero **baja la precision** (0.12–0.15). Es una decisión de diseño (priorizar detectar reingresos) que se refleja directamente en los números: recall relativamente alto, precision baja.

### 7.4 El resultado (la etiqueta) es intrínsecamente ruidoso

Que un paciente reingrese en <30 días vs. en 31 días o más (clase 0) depende de eventos casi aleatorios (una infección, una caída, una decisión del médico de turno). Pacientes con perfiles prácticamente idénticos pueden tener etiquetas distintas, y ningún modelo puede predecir ese ruido.

### 7.5 Decisiones del proyecto que limitan el desempeño (mejorables)

- **Solo 20 columnas** (top 20 por importancia). Se simplifica a cambio de perder algo de señal.
- **Agrupación de categorías:** diagnóstico en 9 grupos y one-hot con "top 10 + otros" descartan detalle (p. ej. el código ICD-9 exacto, la especialidad médica rara).
- **Un solo encuentro por paciente** (el primero): se evita fuga entre train y test, pero se pierde la historia longitudinal del paciente, que suele ser un buen predictor.
- **Submuestra de 20,000 filas** para CV y evaluación de los 4 modelos (por velocidad del SVM). Pero se comprobó que con todo el train el RandomForest casi no mejora (0.2252 → 0.2236), así que esta causa es secundaria.
- **Sin ajuste de hiperparámetros:** los modelos usan configuraciones razonables fijas, sin búsqueda. Normalmente da mejoras pequeñas (unas décimas de punto de AUC), no un salto.
- **Umbral por defecto:** `compute_metrics` usa `predict` (umbral 0.5 en probabilidad). Con un umbral ajustado, el F1 sube solo a ~0.23 (curvas de la sección 3).

### 7.6 Heterogeneidad de los datos

130 hospitales de EE. UU. durante 10 años (1999–2008): prácticas clínicas, codificación y políticas de alta distintas entre sitios. Esa variación mete ruido que el modelo no puede explicar con las columnas disponibles.

### 7.7 Resumen de cómo interpretar los números

| Métrica | Valor | Lectura correcta |
|---|---|---|
| AUC 0.64 | Mejor que azar (0.5), lejos de un buen modelo (≥0.8) | Hay señal débil pero real. |
| Precision 0.145 | ~1.6× la prevalencia (0.09) | De cada 100 pacientes marcados, ~85 **no** reingresan. |
| Recall 0.48–0.50 | Se detecta casi la mitad de los reingresos | Útil como filtro, no como diagnóstico. |
| F1 0.22 | Techo bajo por precision baja | Sube poco al cambiar de modelo o de umbral. |

### 7.8 Uso razonable del modelo

Sirve como **herramienta de priorización**: ordenar pacientes por riesgo y dedicar más seguimiento a los de arriba (en la muestra de la app, los mejor puntuados). No sirve para decidir sobre un paciente individual. Además, el proyecto es educativo.

### 7.9 Qué podría mejorar el desempeño

1. Incorporar datos nuevos con más señal: valores de laboratorio reales, historial longitudinal de múltiples encuentros (cuidando la fuga train/test, por ejemplo dividiendo por paciente), datos post-alta.
2. Entrenar con todos los datos y afinar hiperparámetros (búsqueda en cuadrícula/aleatoria, gradient boosting como XGBoost/LightGBM).
3. Calibrar probabilidades y elegir el umbral con validación según el costo clínico de un falso negativo frente a uno falso positivo.
4. Usar más de 20 columnas o una selección distinta; codificar los diagnósticos con más detalle.

Aun con todo esto, lo esperable es llegar a un AUC de ~0.68–0.70, no a 0.9. **El límite principal es la información, no el modelo.**

---

## 8. Guía rápida para exponer

| Diapositiva | Mensaje en una frase |
|---|---|
| 6 | En validación cruzada, RandomForest tiene el F1 más alto y estable (0.224); las diferencias entre modelos son pequeñas. |
| 7 | En test confirma el patrón: F1 similar entre modelos, y el compromiso precision/recall cambia según el modelo. Las matrices muestran muchas falsas alarmas. |
| 8 | Mover el umbral solo cambia el equilibrio precision/recall; el mejor F1 apenas llega a ~0.23. |
| 9 | Las curvas de los 4 modelos están casi superpuestas (AUC 0.62–0.65): ningún modelo domina. |
| 10 | Se usa F1 de la clase <30 porque el accuracy engaña con 9 % de positivos. |
| 11 | Ganador: RandomForest (F1 0.224, precision 0.145, recall 0.484, AUC 0.648). |
| 12 | Se elige por F1 en CV, estabilidad, recall y simplicidad; la ventaja es modesta. |
