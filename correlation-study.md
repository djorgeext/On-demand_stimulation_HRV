# Informe de metodología experimental: Imputación autorregresiva y pruebas de estrés de robustez bajo pérdida progresiva de intervalos RR

---

## 1. Resumen del estudio

Este estudio evalúa la capacidad de imputación autorregresiva (recursiva) y la resiliencia a la propagación de errores de la red neuronal híbrida entrenada CNN-BiLSTM. Específicamente, el modelo tiene la tarea de reconstruir secuencialmente los ciclos cardíacos faltantes a lo largo de un espectro de severidad de pérdida ($1\%$ al $80\%$).

Dado que cada latido imputado se reincorpora inmediatamente al búfer de contexto histórico para calcular las características futuras de ventana deslizante, este experimento evalúa tanto la precisión de las estimaciones puntuales como la estabilidad dinámica en lazo cerrado a lo largo de horizontes extensos.

```text
                           SERIE RR ORIGINAL
                                   │
                                   ▼
             [Inyección de pérdida aleatoria: 1% - 80%]
            (Preservar búfer de calentamiento: t = 0..39)
                                   │
                                   ▼
                SERIE CORROMPIDA CON LATIDOS FALTANTES
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   │
  Escanear índice t = 40 hasta N-1                           │
  ¿Está RR[t] ausente (NaN)?                                 │
         ├── NO  ──► Conservar el RR[t] real en el historial │
         └── SÍ  ──► PASO DE IMPUTACIÓN AUTORREGRESIVA:      │
                       1. Extraer historial de 40 latidos    │
                       2. Calcular características VFC       │
                       3. Aplicar normalización de 2 niveles │
                       4. Predecir ΔRR mediante CNN-BiLSTM   │
                       5. Reconstruir: RR_hat[t] = ΔRR+RR[t-1]
                       6. Imputar: RR[t] ◄── RR_hat[t] ──────┘
                                   │
                                   ▼
                EVALUACIÓN (RMSE, MAE, R², r de Pearson)

```

---

## 2. Diseño experimental y cohorte de prueba

### 2.1. Cohorte de prueba estratificada

El protocolo de evaluación se ejecutó en $26$ sujetos de prueba diversos que abarcan amplios rangos de edad cronológica y condiciones fisiológicas basales:

$$\mathcal{C}_{\text{test}} = \{\text{16539}, \text{16786}, \text{008}, \text{16420}, \text{16483}, \text{003}, \text{007}, \text{nsr041RRcl}, \text{nsr013RRcl}, \text{nsr033RRcl}, \text{nsr018RRcl}, \text{nsr034RRcl}, \text{nsr003RRcl}, \text{nsr026RRcl}, \text{18177}, \text{16273}, \text{nsr054RRcl}, \text{006}, \text{nsr048RRcl}, \text{19088}, \text{000}, \text{nsr010RRcl}, \text{nsr038RRcl}, \text{nsr022RRcl}, \text{nsr045RRcl}, \text{nsr016RRcl}\}$$

Para cada sujeto, la edad cronológica en años se determina mediante:

$$\text{Edad}_{\text{años}} = \frac{\text{Edad}_{\text{semanas}}}{52.14}$$

y se asigna a su respectivo intervalo de registro experimental $\Omega_k$ en la base de datos de validación (`hrv_validation.h5`).

### 2.2. Cuadrícula de pruebas de estrés

Para capturar la varianza a través de diferentes distribuciones estocásticas de pérdida de datos, la evaluación combina múltiples tasas de eliminación y realizaciones aleatorias independientes:

* **Tasas de pérdida ($P_{\text{elim}}$):** $9$ niveles distintos de degradación:

$$\mathcal{P} = \{0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80\} \quad (1\% \text{ al } 80\%)$$

* **Semillas de Monte Carlo ($\mathcal{S}$):** $5$ generadores aleatorios independientes para distribuciones espaciales reproducibles de latidos faltantes:

$$\mathcal{S} = \{7, 101, 211, 317, 421\}$$

* **Total de ensayos de simulación:**

$$N_{\text{ensayos}} = 12 \text{ sujetos} \times 9 \text{ tasas} \times 5 \text{ semillas} = 540 \text{ ejecuciones de simulación independientes}$$

---

## 3. Protocolo de inyección de pérdidas de datos

Sea la serie completa de referencia (*ground-truth*) de longitud $N$ denotada por $S = [RR_0, RR_1, \dots, RR_{N-1}]$.

```text
Índice:   0               39  40                                         N-1
Serie:   [■ ■ ■ ■ ... ■ ■ ■ ][□ ■ □ □ ■ ■ □ ■ ■ □ ... ■ □ ■ ■ □ ■]
         |◄─ Calentamiento ─►||◄───── Ventana de pérdida aleatoria ──────►|
          (Sin NaNs permitidos)  n_elim = round(N · P_elim) posiciones como NaN

```

### Reserva del contexto de calentamiento (*warm-up*):

Para inicializar la extracción de características sin necesidad de rellenar bordes de forma artificial (*edge padding*), los primeros $W_{\text{long}} = 40$ latidos ($t \in [0, 39]$) quedan protegidos contra eliminación:

$$S_{\text{corrupted}}[t] = S[t], \quad \forall t < 40$$

### Muestreo aleatorio uniforme sin reemplazo:

El número total de latidos seleccionados para eliminación es:

$$n_{\text{elim}} = \left\lfloor N \cdot P_{\text{elim}} \right\rceil$$

Se extrae un subconjunto aleatorio de índices $\mathcal{I}_{\text{elim}} \subset \{40, 41, \dots, N-1\}$ con cardinalidad $\vert{}\mathcal{I}_{\text{elim}}\vert{} = n_{\text{elim}}$ mediante muestreo uniforme sin reemplazo utilizando la semilla activa $s \in \mathcal{S}$:

$$S_{\text{corrupted}}[t] = \begin{cases} \text{NaN} & \text{si } t \in \mathcal{I}_{\text{elim}} \\ S[t] & \text{si } t \notin \mathcal{I}_{\text{elim}} \end{cases}$$

---

## 4. Motor de imputación autorregresiva paso a paso

El proceso de recuperación escanea secuencialmente desde $t = 40$ hasta $N - 1$. Al encontrar un valor faltante ($S_{\text{corrupted}}[t] = \text{NaN}$), el modelo ejecuta el siguiente flujo de inferencia:

```text
[t-40 : t] Búfer histórico ──► Extraer características VFC ──► Escalamiento en 2 niveles
                                                                     │
                                                                     ▼
Latido reconstruido: ◄── Desescalado y offset ◄── Predicción rápida (GPU)
RR_hat[t] = ΔRR + RR[t-1]  ΔRR = y_norm · σ_y + μ_y       [CNN-BiLSTM]
        │
        ▼
Imputar en S_corrupted[t] ──► Disponible como referencia para t+1, t+2...

```

### Paso 1: Extracción del contexto histórico

La ventana precedente de 40 latidos se recupera del arreglo actualizado dinámicamente:

$$\mathbf{h}_t = S_{\text{corrupted}}[t-40 : t] = \big[ \widetilde{RR}_{t-40}, \widetilde{RR}_{t-39}, \dots, \widetilde{RR}_{t-1} \big]^\top \in \mathbb{R}^{40}$$

*Nota: Si algún latido dentro de $[t-40, t-1]$ estuvo ausente originalmente, $\mathbf{h}_t$ contendrá los valores previamente predichos, creando un proceso autorregresivo genuino en lazo cerrado.*

### Paso 2: Extracción dinámica de características

Se pasa un segmento sintético de evaluación $\mathbf{w}_t = [\mathbf{h}_t \,\Vert{}\, \text{NaN}] \in \mathbb{R}^{41}$ al extractor de características:

* **Secuencia a corto plazo (20 latidos):**

$$\mathbf{X}_{\text{rr}} = \big[ \widetilde{RR}_{t-20}, \widetilde{RR}_{t-19}, \dots, \widetilde{RR}_{t-1} \big] \in \mathbb{R}^{1 \times 20}$$

* **Métricas estadísticas y no lineales de VFC (8 características):**

$$\mathbf{X}_{\text{feat}} = \big[ \mu_{RR}, SDSD, SD_2, CCM, Guzik, NN50, Porta, STD \big] \in \mathbb{R}^{1 \times 8}$$

### Paso 3: Normalización exacta en dos niveles

Utilizando la edad basal del sujeto $A$ y los parámetros poblacionales del intervalo obtenidos de `/normalization/{interval}`:

* **Escalamiento por referencia de edad:**

$$\mu_{RR}^{\text{ref}}(A) = 505 \cdot A^{0.122}, \quad \sigma_{RR}^{\text{ref}}(A) = \begin{cases} 80 \cdot A^{0.26} & \text{si } A \le 12 \\ 290 \cdot A^{-0.20} & \text{si } A > 12 \end{cases}$$

$$\mathbf{X}_{\text{rr}}^{\text{norm}} = \frac{\mathbf{X}_{\text{rr}} - \mu_{RR}^{\text{ref}}(A)}{\sigma_{RR}^{\text{ref}}(A)} \in \mathbb{R}^{1 \times 20 \times 1}$$

* **Estandarización poblacional por intervalo:**

$$\mathbf{X}_{\text{feat}}^{\text{norm}} = \frac{\mathbf{X}_{\text{feat}} - \boldsymbol{\mu}_{\text{feat}}^{(k)}}{\boldsymbol{\sigma}_{\text{feat}}^{(k)}} \in \mathbb{R}^{1 \times 8}$$

### Paso 4: Inferencia acelerada de alto rendimiento (`fast_predict`)

El método estándar `model.predict()` genera sobrecarga de trazado de grafos en inferencias de una sola instancia. Para lograr una ejecución rápida a lo largo de millones de iteraciones, la inferencia se ejecuta como un grafo computacional compilado:

$$\widehat{y}_{\text{norm}} = \mathcal{M}_{\boldsymbol{\theta}}\Big( \mathbf{X}_{\text{rr}}^{\text{norm}}, \mathbf{X}_{\text{feat}}^{\text{norm}} ; \text{training}=\text{False} \Big)$$

ejecutado bajo `@tf.function(reduce_retracing=True)` en hardware GPU.

### Paso 5: Desnormalización y reconstrucción absoluta del latido

La red produce el delta estandarizado $\widehat{y}_{\text{norm}}$. La diferencia física en milisegundos ($\widehat{\Delta RR}$) y el latido absoluto reconstruido ($\widehat{RR}_t$) se obtienen mediante:

$$\widehat{\Delta RR}_{\text{ms}} = \big( \widehat{y}_{\text{norm}} \cdot \sigma_{\text{target}}^{(k)} \big) + \mu_{\text{target}}^{(k)}$$

$$\widehat{RR}_t = \widehat{\Delta RR}_{\text{ms}} + S_{\text{corrupted}}[t-1]$$

donde $S_{\text{corrupted}}[t-1]$ es el intervalo inmediatamente anterior (real o imputado).

### Paso 6: Actualización del estado autorregresivo

El valor reconstruido se escribe directamente en el arreglo de trabajo:

$$S_{\text{corrupted}}[t] \leftarrow \widehat{RR}_t$$

Este valor actualizado influye de inmediato en los promedios móviles, desviaciones estándar, coordenadas de Poincaré y secuencias autorregresivas para todos los puntos subsiguientes $t' > t$.

---

## 5. Marco de evaluación y métricas cuantitativas

El rendimiento se mide en dos niveles: error de imputación puntual local en los latidos ausentes y preservación morfológica global a lo largo de toda la señal reconstruida.

```text
Puntos temporales evaluados:
  Serie original:     ──[RR_0]──[RR_1]── ... ──[RR_i]── ... ──[RR_N-1]──
  Corrompida (NaNs):  ──[RR_0]──[RR_1]── ... ──[ NaN]── ... ──[RR_N-1]──
  Reconstruida:       ──[RR_0]──[RR_1]── ... ──[RR̂_i]── ... ──[RR_N-1]──
                                                  ▲
                         Métricas a nivel de punto (i ∈ I_elim):
                         • RMSE, MAE, R²
                         Métrica global de señal (t = 0..N-1):
                         • Correlación de Pearson r(S_orig, S_filled)

```

### 5.1. Métricas de imputación a nivel de punto (evaluadas en $i \in \mathcal{I}_{\text{elim}}$)

* **Raíz del error cuadrático medio ($\text{RMSE}$):**

$$\text{RMSE} = \sqrt{\frac{1}{\vert{}\mathcal{I}_{\text{elim}}\vert{}} \sum_{i \in \mathcal{I}_{\text{elim}}} \left( RR_i^{\text{true}} - \widehat{RR}_i \right)^2}$$

* **Error absoluto medio ($\text{MAE}$):**

$$\text{MAE} = \frac{1}{\vert{}\mathcal{I}_{\text{elim}}\vert{}} \sum_{i \in \mathcal{I}_{\text{elim}}} \left\vert{} RR_i^{\text{true}} - \widehat{RR}_i \right\vert{}$$

* **Coeficiente de determinación ($R^2$):**

$$R^2 = 1 - \frac{\sum_{i \in \mathcal{I}_{\text{elim}}} \left( RR_i^{\text{true}} - \widehat{RR}_i \right)^2}{\sum_{i \in \mathcal{I}_{\text{elim}}} \left( RR_i^{\text{true}} - \overline{RR}_{\text{elim}}^{\text{true}} \right)^2}$$

### 5.2. Métrica de preservación global de la señal

* **Coeficiente de correlación de Pearson ($r$):**
Mide la preservación del ritmo global, la distribución de la potencia espectral y las derivas basales de baja frecuencia en toda la serie temporal reconstruida $S_{\text{filled}}$ en comparación con la serie no corrompida $S_{\text{orig}}$ ($N$ latidos):

$$r = \frac{\sum_{t=0}^{N-1} (RR_t^{\text{orig}} - \overline{RR}^{\text{orig}})(RR_t^{\text{filled}} - \overline{RR}^{\text{filled}})}{\sqrt{\sum_{t=0}^{N-1} (RR_t^{\text{orig}} - \overline{RR}^{\text{orig}})^2} \sqrt{\sum_{t=0}^{N-1} (RR_t^{\text{filled}} - \overline{RR}^{\text{filled}})^2}}$$

---