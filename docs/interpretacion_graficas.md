# Interpretación de las gráficas de comparación de modelos

Este documento explica cuatro figuras de `reports/figures/`: qué muestran, cómo se generan y cómo leerlas. Al final hay conclusiones.

**Contexto.** Se predice si un paciente diabético reingresa en menos de 30 días (`<30`). Solo ~11,4 % de los casos son positivos, así que el problema está muy **desbalanceado**. Los clasificadores son GaussianNB, DecisionTree, RandomForest y SVM. Los regresores son DecisionTreeRegressor, RandomForestRegressor y SVR, sobre el target ordinal `NO=0`, `>30=1`, `<30=2`.

Todas las funciones de graficado están en `src/evaluation.py`. Se ejecutan desde `train.py` (pipeline completo) y desde los notebooks `03_modelado` y `04_evaluacion`.

---

## 1. `cv_boxplot.png` — F1 por modelo en validación cruzada

**Cómo se genera**

- `ev.plot_cv_boxplot(cv_results)` (`src/evaluation.py`).
- `cv_results` sale de `cross_validate_models` (`src/models.py`). Usa `StratifiedGroupKFold` con 5 folds sobre la submuestra de ~20 000 filas de train.
- Los folds se agrupan por paciente (`patient_nbr`), así que un mismo paciente nunca queda en train y validación a la vez.
- Cada modelo produce 5 valores de F1 de la clase `<30`, y cada caja resume esos 5 valores.

**Cómo leerla**

- Eje X: modelo. Eje Y: F1.
- La línea central de la caja es la mediana. La caja cubre el rango intercuartílico, y los bigotes llegan al mínimo y máximo (sin atípicos).
- **Altura de la caja** = rendimiento típico. **Tamaño de la caja** = estabilidad: una caja pequeña da resultados parecidos entre folds.
- **Solapamiento de cajas**: si se solapan, la diferencia entre esos modelos puede ser ruido.

**Valores reales** (`artifacts/cv_results.csv`)

| Modelo | F1 medio | Desv. | Mín | Máx |
|---|---|---|---|---|
| RandomForest | 0,273 | 0,008 | 0,262 | 0,284 |
| DecisionTree | 0,266 | 0,010 | 0,250 | 0,277 |
| SVM | 0,260 | 0,013 | 0,239 | 0,271 |
| GaussianNB | 0,213 | 0,032 | 0,178 | 0,255 |

**Lectura.** RandomForest, DecisionTree y SVM quedan en un rango muy estrecho (0,26–0,27) y sus cajas se solapan. GaussianNB queda claramente más abajo y es el más inestable (caja más alta y bigotes más largos). Este gráfico **no usa el set de test**, por eso es el criterio oficial para elegir ganador (ADR-03).

---

## 2. `comparativa_metricas.png` — Métricas de clasificación en test

**Cómo se genera**

- `ev.compute_metrics` entrena nada nuevo: usa cada modelo ya ajustado y calcula `accuracy`, `precision`, `recall` y `F1` (clase `<30`) con `model.predict` sobre el **test** (19 870 encuentros, pacientes que el modelo no vio).
- `ev.plot_metrics_bars(metrics)` pasa la tabla a formato largo y dibuja barras agrupadas con seaborn: una barra por modelo dentro de cada métrica.
- Se hace en `train.py` (paso 6) y en `04_evaluacion` (celda de barras).

**Cómo leerla**

- Cada grupo de barras es una métrica y cada color un modelo.
- **accuracy**: porcentaje de aciertos totales. Con 11 % de positivos engaña: predecir "nadie reingresa" ya da ~89 %.
- **precision**: de los que el modelo marca como reingreso, cuántos lo son de verdad. Baja precision significa muchas falsas alarmas.
- **recall**: de los que sí reingresan, cuántos detecta el modelo. Un recall bajo significa que se escapan casos.
- **F1**: media armónica de precision y recall. Es la métrica que resume el compromiso y la que se usa para decidir.

**Valores reales** (`artifacts/metrics.csv`)

| Modelo | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| GaussianNB | 0,838 | 0,234 | 0,186 | 0,207 |
| RandomForest | 0,684 | 0,180 | 0,499 | 0,265 |
| DecisionTree | 0,654 | 0,167 | 0,510 | 0,251 |
| SVM | 0,654 | 0,174 | 0,543 | 0,264 |

**Lectura.**

- GaussianNB tiene la mejor accuracy (0,84) pero el peor F1 y detecta solo ~19 % de los reingresos. Es un caso típico de accuracy engañosa: acierta mucho porque casi siempre predice "no reingresa".
- Los otros tres usan `class_weight="balanced"`. Sacrifican accuracy y precision, pero detectan cerca de la mitad de los reingresos (recall ≈ 0,50–0,54).
- RandomForest y SVM prácticamente empatan en F1 (0,265 vs 0,264).

---

## 3. `comparativa_pr.png` — Curvas precision-recall

**Cómo se genera**

- `ev.risk_score` da un puntaje continuo por paciente: `predict_proba` si el modelo lo tiene, y para SVM la `decision_function` pasada por una sigmoide (se entrenó con `probability=False` por velocidad).
- `ev.plot_comparison(all_scores, y_test)` llama a `sklearn.metrics.precision_recall_curve(y_test, scores)` para cada modelo. Esa función barre todos los umbrales de decisión posibles y devuelve el par (precision, recall) de cada uno.
- Se dibuja recall en X y precision en Y, una línea por modelo. La misma función también produce `comparativa_umbral_f1.png`.

**Cómo leerla**

- Cada punto de la curva es un **umbral** distinto. Moverse a la derecha equivale a marcar a más pacientes como riesgo: sube el recall y baja la precision.
- Una curva **más arriba y a la derecha** es mejor, porque da más precision para un mismo recall.
- La línea de referencia de un modelo al azar es horizontal a la **prevalencia** (~0,114). Aquí es donde terminan todas las curvas en recall = 1.
- A diferencia de las barras, esta figura **no depende de un umbral fijo**: compara la capacidad de ordenar pacientes por riesgo.

**Lectura.**

- Las curvas están casi superpuestas. Pasan de ~0,3 de precision con recall bajo a ~0,2 con recall 0,4 y ~0,12 con recall 1.
- Las diferencias solo aparecen a recall muy bajo (< 0,1), donde hay pocos pacientes y la curva es ruidosa. DecisionTree se ve como líneas rectas porque, con `max_depth=3`, solo produce unos pocos puntajes distintos.
- La precision máxima útil ronda 0,2–0,3, apenas 2 a 3 veces la prevalencia. **Ningún modelo separa bien los casos**: la señal en los datos es débil.

---

## 4. `comparativa_regresion.png` — MSE / MAE / RMSE de los regresores

**Cómo se genera**

- Cada regresor se ajusta con `tune_model(..., regression=True)` y se predice el target ordinal (0, 1, 2) sobre el test.
- `ev.compute_regression_metrics` calcula `MSE`, `MAE` y `RMSE = √MSE`. `ev.plot_regression_bars` hace barras agrupadas como en la figura 2.

**Cómo leerla**

- **Menor es mejor** en las tres métricas (al revés que en clasificación).
- **MAE**: error medio absoluto, en "niveles" del target. Un MAE de 0,56 significa que se falla en promedio medio nivel.
- **MSE / RMSE**: elevan el error al cuadrado, así que castigan más los fallos grandes (predecir 0 cuando era 2). RMSE tiene las mismas unidades que el target.
- Si MAE es bajo pero RMSE es alto, hay pocos errores muy grandes.

**Valores reales** (`artifacts/metrics_regression.csv`)

| Modelo | MSE | MAE | RMSE |
|---|---|---|---|
| RandomForestRegressor | 0,433 | 0,563 | **0,658** |
| DecisionTreeRegressor | 0,440 | 0,573 | 0,663 |
| SVR | 0,490 | 0,521 | 0,700 |

**Lectura.**

- RandomForestRegressor gana por RMSE (criterio de ADR-03), pero su ventaja sobre el árbol es mínima.
- SVR tiene el **mejor MAE** (0,52) y el **peor RMSE** (0,70). Suele acertar más de cerca, pero comete algunos errores grandes que el cuadrado penaliza. Cambiar la métrica de selección podría cambiar el ganador.
- Para saber si estos números son buenos hay que compararlos con la línea base de predecir siempre la media. El notebook 03 (celda final) la calcula, pero la celda no tiene salida guardada. Vuelve a ejecutarla para tener la cifra.

---

## Conclusiones

1. **El problema es difícil.** Con F1 ≈ 0,27 y precision ≈ 0,18, solo ~1 de cada 5 alertas es un reingreso real. Los datos disponibles predicen poco el reingreso a 30 días.
2. **La accuracy no sirve aquí.** GaussianNB la lidera (0,84) y es el peor modelo para lo que importa. Hay que mirar F1 y recall.
3. **Ganador de clasificación: RandomForest.** Tiene el mejor F1 medio en CV (0,273) y el más estable (desv. 0,008). Pero su ventaja sobre DecisionTree y SVM es pequeña y las cajas del boxplot se solapan. Elegirlo es razonable, no concluyente.
4. **Hay un compromiso que se puede ajustar.** RandomForest y SVM detectan ~50–54 % de los reingresos pero con muchas falsas alarmas. Si el costo de no detectar un reingreso es mayor que el de una alerta de más, convendría bajar el umbral y priorizar recall (ver `umbral_*.png`).
5. **Las curvas PR confirman un techo común.** Si cuatro algoritmos muy distintos dan casi la misma curva, el límite lo pone la información de los datos y no el modelo. Más tuning probablemente aporte poco. Mejorarían más nuevas variables o más datos.
6. **Regresión: RandomForestRegressor gana por RMSE, pero por muy poco.** SVR gana por MAE, así que la elección depende de si se quiere penalizar los errores grandes.
7. **Cautela.** Los modelos se entrenaron con una submuestra de ~20 000 filas (por la lentitud de SVM/SVR), y estas diferencias son del orden de 0,01 de F1. Con otra semilla o con todo el train, el orden entre RandomForest, SVM y DecisionTree podría cambiar.
