# Formulación Matemática del Pipeline de Preprocesamiento y Extracción de Características

---

### 1. Representación de la Señal Cruda y Ventaneo Temporal

Sea el ritmo cardíaco de un sujeto representado como una secuencia discreta de intervalos entre latidos (intervalos RR):

$$S = \{RR_1, RR_2, \dots, RR_M\}, \quad RR_t \in \mathbb{Z}^+$$

donde cada $RR_t$ denota la duración transcurrida entre dos picos consecutivos de la onda R en milisegundos.

Para capturar la dinámica autonómica local sin asumir estacionariedad a largo plazo, la serie continua se segmenta mediante un esquema de ventaneo deslizante de dos escalas temporales:

1. **Ventana de Análisis Corta ($W_{\text{corta}} = 20$ latidos):** Captura la trayectoria temporal instantánea, la tendencia central del período cardíaco local y las transiciones latido a latido:

$$\mathbf{w}_{\text{corta}}(t) = \big[ RR_{t-19}, RR_{t-18}, \dots, RR_t \big]^\top \in \mathbb{R}^{20}$$



Las componentes individuales de este vector definen las características de la trayectoria instantánea:

$$rr_i(t) = RR_{t - 20 + i}, \quad \text{para } i \in \{1, 2, \dots, 20\}$$


2. **Ventana de Análisis Larga ($W_{\text{larga}} = 40$ latidos):** Proporciona un contexto temporal más amplio para estimar la dispersión de varianza, las métricas no lineales en el espacio de fases y la asimetría de la distribución sobre un estado cuasi-estacionario:

$$\mathbf{w}_{\text{larga}}(t) = \big[ RR_{t-39}, RR_{t-38}, \dots, RR_t \big]^\top \in \mathbb{R}^{40}$$



---

### 2. Extracción de Características y Formulaciones Matemáticas

A partir de cada segmento ventaneado, se extrae un vector predictor de 28 dimensiones $\mathbf{x} \in \mathbb{R}^{28}$. Las características se dividen en dos subconjuntos operativos según su escala y naturaleza fisiológica:

```
                      Secuencia Ventaneada de Intervalos RR
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
[Subconjunto Específico por Sujeto (21)]            [Subconjunto Normalizado por Intervalo (7)]
  • Período cardíaco medio: μ_RR                     • Dominio temporal: SDSD, STD, NN50
  • Secuencia de latidos: rr_1 ... rr_20             • Espacio de fases / No lineal: SD_2, CCM
                                                     • Asimetría de frecuencia cardíaca: Guzik, Porta

```

#### 2.1. Subconjunto Temporal Específico por Sujeto (21 Características)

* **Período Cardíaco Medio Local ($\mu_{RR}$):**

$$\mu_{RR}(t) = \frac{1}{W_{\text{corta}}} \sum_{i=1}^{W_{\text{corta}}} rr_i(t) = \frac{1}{20} \sum_{i=1}^{20} rr_i(t)$$


* **Secuencia Temporal de Latidos ($rr_1, \dots, rr_{20}$):**
Secuencia ordenada $\mathbf{w}_{\text{corta}}(t)$ que describe la trayectoria directa entre latidos sucesivos.

---

#### 2.2. Subconjunto Estadístico, Geométrico y No Lineal de VFC (7 Características)

Sea $\Delta RR_i = RR_{i+1} - RR_i$ el operador de diferencia regresiva de primer orden sobre una ventana de análisis de longitud $W$.

* **Desviación Estándar de Diferencias Sucesivas ($SDSD$):**
Cuantifica las fluctuaciones autonómicas de alta frecuencia a corto plazo:

$$SDSD = \sqrt{\frac{1}{W - 2} \sum_{i=1}^{W-1} \left( \Delta RR_i - \overline{\Delta RR} \right)^2}$$



donde $\overline{\Delta RR} = \frac{1}{W-1} \sum_{i=1}^{W-1} \Delta RR_i$.
* **Desviación Estándar Muestral ($STD$ / $SDRR$):**
Refleja la dispersión cíclica total y la potencia autonómica global:

$$STD = \sqrt{\frac{1}{W - 1} \sum_{i=1}^{W} \left( RR_i - \mu_{RR} \right)^2}$$


* **Conteo de Superación de Umbral ($NN50$):**
Cuantifica variaciones transitorias de mediación predominantemente parasimpática:

$$NN50 = \sum_{i=1}^{W-1} \mathbb{I}\left( \vert{}\Delta RR_i\vert{} > 50 \text{ ms} \right)$$



donde $\mathbb{I}(\cdot)$ es la función indicadora.
* **Semieje Mayor del Diagrama de Poincaré ($SD_2$):**
En el espacio de fases bidimensional (mapa de retorno) $(RR_i, RR_{i+1})$, una rotación de coordenadas a $45^\circ$ proyecta los puntos sobre ejes ortogonales:

$$x_1(i) = \frac{RR_i - RR_{i+1}}{\sqrt{2}}, \quad x_2(i) = \frac{RR_i + RR_{i+1}}{\sqrt{2}}$$



$SD_2$ mide la dispersión continua a lo largo de la línea de identidad ($x_2$), asociada a modulaciones autonómicas de baja frecuencia y a la varianza global:

$$SD_2 = \sqrt{\text{Var}(x_2)} = \sqrt{2 \cdot \text{Var}(RR) - \frac{1}{2} \text{Var}(\Delta RR)}$$


* **Medida de Correlación Compleja ($CCM$):**
Cuantifica la variación punto a punto de la trayectoria temporal entre tripletes consecutivos en el plano de Poincaré. Para tres puntos consecutivos $p_{i-1}, p_i, p_{i+1}$, el área con signo del triángulo encerrado es:

$$A_i = \frac{1}{2} \det \begin{pmatrix} RR_{i-1} & RR_i & 1 \\ RR_i & RR_{i+1} & 1 \\ RR_{i+1} & RR_{i+2} & 1 \end{pmatrix}$$



Normalizando respecto al área de la elipse envolvente $\pi \cdot SD_1 \cdot SD_2$ (donde $SD_1 = \sqrt{\frac{1}{2}\text{Var}(\Delta RR)}$):

$$CCM = \frac{1}{(W - 2) \cdot \pi \cdot SD_1 \cdot SD_2} \sum_{i=1}^{W-2} \vert{}A_i\vert{}$$


* **Índices de Asimetría de la Frecuencia Cardíaca ($Porta$ y $Guzik$):**
Cuantifican la contribución diferencial de las desaceleraciones cardíacas ($\Delta RR_i > 0$) frente a las aceleraciones ($\Delta RR_i < 0$), producto de la histéresis simpático-vagal no lineal:
$$\text{Índice de Porta: } Porta = \frac{N(\Delta RR_i < 0)}{N(\Delta RR_i \neq 0)}$$


$$\text{Índice de Guzik: } Guzik = \frac{\sum_{i: \Delta RR_i > 0} (\Delta RR_i)^2}{\sum_{i=1}^{W-1} (\Delta RR_i)^2}$$



---

### 3. Estrategia de Normalización en Dos Niveles

Para neutralizar variables de confusión inter-sujeto y preservar la discriminabilidad de los estados autonómicos, la normalización se organiza en dos niveles:

```
                            Características Extraídas (28D)
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────┐
        ▼                                                                   ▼
[Nivel 1: Específico por Sujeto]                            [Nivel 2: Métricas Derivadas de VFC]
Variables: {μ_RR, rr_1 ... rr_20}                           Variables: {SDSD, SD_2, CCM, Guzik,
                                                                        NN50, Porta, STD}
        │                                                                   │
        ▼                                                                   ▼
Cálculo de Edad:                                            Extracción de Estadísticos de Intervalo:
  A = Edad_semanas / 52.14                                    μ_intervalo, σ_intervalo
        │                                                                   │
        ▼                                                                   ▼
Modelos Paramétricos de Referencia:                         Puntuación Z Poblacional Estándar:
  μ_ref(A) = 505 · A^(0.122)                                  z = (x - μ_intervalo) / σ_intervalo
  σ_ref(A) = modelo a trozos (A ≤ 12 vs A > 12)
        │
        ▼
Estandarización Fisiológica:
  z = (x - μ_ref(A)) / σ_ref(A)

```

---

#### 3.1. Nivel 1: Escalamiento Paramétrico Dependiente de la Edad Fisiológica

La frecuencia cardíaca basal y la variabilidad del período cardíaco evolucionan a lo largo del desarrollo ontogénico: el período cardíaco en reposo se incrementa asintóticamente con la edad (disminución de la frecuencia cardíaca basal), mientras que la variabilidad basal se expande durante el desarrollo pediátrico antes de iniciar un declive gradual en la madurez.

1. **Conversión de Edad Cronológica:**

$$A = \frac{\text{Edad}_{\text{semanas}}}{52.14}$$


2. **Modelos de Expectativa Normativa:**
La media de referencia basal $\mu_{RR}^{\text{ref}}(A)$ y la dispersión de referencia $\sigma_{RR}^{\text{ref}}(A)$ (ambas en milisegundos) se determinan mediante modelos empíricos de escalamiento alométrico:
$$\mu_{RR}^{\text{ref}}(A) = 505 \cdot A^{0.122}$$


$$\sigma_{RR}^{\text{ref}}(A) = \begin{cases} 80 \cdot A^{0.26} & \text{si } A \le 12 \text{ años (etapa de desarrollo pediátrico)} \\ 290 \cdot A^{-0.20} & \text{si } A > 12 \text{ años (etapa de declive en adultos)} \end{cases}$$


3. **Mapeo de Estandarización:**
Para toda característica $u \in \{\mu_{RR}, rr_1, rr_2, \dots, rr_{20}\}$ de un sujeto de edad $A$:

$$\widetilde{u} = \frac{u - \mu_{RR}^{\text{ref}}(A)}{\max\left(\sigma_{RR}^{\text{ref}}(A), \epsilon\right)}, \quad \epsilon = 10^{-8}$$



Esta transformación desvincula la maduración autonómica debida a la edad de los patrones fisiopatológicos o estados experimentales agudos.

---

#### 3.2. Nivel 2: Estandarización Z a Nivel Poblacional del Intervalo

Las métricas derivadas ($SDSD, SD_2, CCM, Guzik, NN50, Porta, STD$) poseen distintas unidades y escalas numéricas. Estas características se estandarizan contra los momentos empíricos del conjunto de entrenamiento dentro de un intervalo experimental o condición de registro $\Omega_k$.

Para cada característica $v \in \mathcal{F}_{\text{norm}}$:

$$\widetilde{v}^{(k)} = \frac{v - \mu_v^{(k)}}{\sigma_v^{(k)}}$$

donde $\mu_v^{(k)}$ y $\sigma_v^{(k)}$ representan la media y la desviación estándar muestral calculadas sobre la totalidad de muestras del intervalo:

$$\mu_v^{(k)} = \frac{1}{N_k} \sum_{m \in \Omega_k} v_m, \quad \sigma_v^{(k)} = \sqrt{\frac{1}{N_k - 1} \sum_{m \in \Omega_k} \left( v_m - \mu_v^{(k)} \right)^2}$$

Para asegurar estabilidad numérica frente a dispersiones nulas o cuasi-constantes:

$$\sigma_v^{(k)} \leftarrow \begin{cases} \sigma_v^{(k)} & \text{si } \sigma_v^{(k)} \ge \epsilon \\ 1.0 & \text{si } \sigma_v^{(k)} < \epsilon \end{cases}, \quad \epsilon = 10^{-8}$$

---

### 4. Algoritmo de Estimación de Momentos en Dos Pasadas

Para estimar los momentos poblacionales exactos en conjuntos de datos masivos sin desbordar la memoria de trabajo, el pipeline emplea un algoritmo en flujo de dos pasadas:

```
PASADA 1: Acumulación de Momentos en Flujo (Memoria O(1))
══════════════════════════════════════════════════════════════════════════
Para cada sujeto s en el intervalo k:
  1. Extraer características ventaneadas: X_s ∈ ℝ^(n_s × 7)
  2. Acumular:
       S_1 ← S_1 + Σ_{m=1}^{n_s} X_{s, m}
       S_2 ← S_2 + Σ_{m=1}^{n_s} (X_{s, m} ⊙ X_{s, m})
       N_k ← N_k + n_s

Finalizar Estadísticos Globales del Intervalo (ddof = 1):
  μ^(k)   = (1 / N_k) · S_1
  (σ^(k))² = (1 / (N_k - 1)) · [ S_2 - (1 / N_k) · (S_1 ⊙ S_1) ]
  σ^(k)   = sqrt( max( (σ^(k))², 0 ) )

PASADA 2: Transformación Progresiva y Serialización
══════════════════════════════════════════════════════════════════════════
Para cada sujeto s en el intervalo k:
  1. Re-extraer características ventaneadas
  2. Aplicar escalamiento Nivel 1 a {μ_RR, rr_1 ... rr_20} según la edad A_s
  3. Aplicar estandarización Nivel 2 a {SDSD ... STD} según μ^(k), σ^(k)
  4. Conformar la matriz estandarizada X_s ∈ ℝ^(n_s × 28) (float32)
  5. Escribir X_s y el vector de etiquetas y_s en almacenamiento jerárquico persistente

```

---

### 5. Representación Final del Conjunto de Datos y Lógica de Indexación

#### 5.1. Conformación del Vector de Entrada

Para cada muestra deslizante $m$, el vector de entrada estandarizado se ensambla como:

$$\mathbf{x}_m = \Big[ \widetilde{sdsd}_m, \widetilde{sd2}_m, \widetilde{ccm}_m, \widetilde{guzik}_m, \widetilde{nn50}_m, \widetilde{porta}_m, \widetilde{std}_m, \widetilde{\mu}_{RR, m}, \widetilde{rr}_{1, m}, \dots, \widetilde{rr}_{20, m} \Big]^\top \in \mathbb{R}^{28}$$

junto con su etiqueta objetivo $y_m \in \mathcal{Y}$.

---

#### 5.2. Muestreo Proporcional y Lógica de Indexación

Considerando un conjunto de datos distribuido en $K$ intervalos experimentales con $S_k$ sujetos por intervalo, cada sujeto $s$ aporta un bloque de datos:

$$\mathbf{X}_s \in \mathbb{R}^{n_s \times 28}, \quad \mathbf{y}_s \in \mathbb{R}^{n_s}$$

El volumen total de muestras del estudio es:

$$N_{\text{total}} = \sum_{k=1}^K \sum_{s=1}^{S_k} n_s$$

Para evitar sesgos derivados de diferencias en la duración de los registros ($n_s$), se estructura una tabla de índice maestro que almacena la tupla:

$$\mathcal{I}_s = \Big( \text{Intervalo}_k, \text{ID\_Sujeto}_s, \text{Ruta\_Almacenamiento}_s, n_s, A_s \Big)$$

Esta estructura habilita dos esquemas de muestreo estocástico durante el entrenamiento de modelos:

1. **Muestreo Uniforme por Ventana:** La probabilidad de seleccionar una ventana del sujeto $s$ es proporcional a su volumen de muestras:

$$P(\text{Sujeto } s) = \frac{n_s}{N_{\text{total}}}$$


2. **Estratificación Balanceada por Sujeto:** Asigna probabilidad uniforme a cada sujeto independientemente de la duración de su registro:

$$P(\text{Sujeto } s) = \frac{1}{\sum_{k=1}^K S_k}$$



evitando que sujetos con registros extensos dominen la función de pérdida del modelo.

---