"""
tools/glossary_es.py - Spanish meanings of the acronyms in tools/glossary.py, for the Spanish PDFs
(tools/make_architecture_pdfs_es.py). The terms, match modes and the undefined-acronym check stay in
glossary.py; this file only translates. A term added there without a translation here fails the import.
"""
import unicodedata

import glossary as G

MEANING_ES = {
    "1D": "unidimensional (capa que se desliza solo a lo largo del eje temporal, p. ej. Conv1D)",
    "AdamW": "optimizador Adam con decaimiento de pesos desacoplado (algoritmo de entrenamiento de los modelos neuronales)",
    "AR": "autorregresivo: un valor predicho como suma ponderada de valores pasados; AR(p) usa p valores pasados",
    "Burg-AR": "modelo AR cuyos coeficientes se ajustaron con el método de Burg (usado en los primeros 30 latidos)",
    "ccm": "medida de correlación compleja: índice de no linealidad del diagrama de Poincaré de intervalos RR consecutivos",
    "CCM": "medida de correlación compleja (ver ccm)",
    "CMSIS-NN": "Cortex Microcontroller Software Interface Standard - Neural Network: biblioteca optimizada de Arm "
                "para redes neuronales en microcontroladores Cortex-M",
    "Conv1D": "capa de convolución unidimensional",
    "Cortex-M7": "Arm Cortex-M7, el núcleo de procesador del microcontrolador STM32H750",
    "CPU": "unidad central de procesamiento",
    "CSV": "valores separados por comas (archivo de tabla en texto plano)",
    "DAD": "data as demonstrator: entrenamiento con entradas que contienen los rellenos de lazo cerrado del propio "
           "modelo, con el objetivo verdadero (remedio contra el sesgo de exposición)",
    "_dad": "sufijo de nombre de ejecución: ajuste fino con datos DAD (rellenados en lazo cerrado)",
    "DC": "corriente continua, es decir, la componente constante (frecuencia cero) de una señal",
    "dRR": "cambio latido a latido del intervalo RR: dRR[n+1] = RR[n+1] - RR[n] (ms); la magnitud que se predice",
    "ECG": "electrocardiograma",
    "EMA": "media móvil exponencial",
    "ema_mean": "referencia que predice el siguiente RR como media móvil exponencial de la ventana",
    "_f8": "sufijo de nombre de ejecución: 8 filtros de convolución en lugar de 16",
    "FLOPs": "operaciones de coma flotante (coste aritmético)",
    "_ft": "sufijo de nombre de ejecución: ajuste fino (entrenamiento continuado desde los pesos del modelo padre)",
    "GBDT": "árboles de decisión potenciados por gradiente (gradient-boosted decision trees)",
    "gbdt": "árboles de decisión potenciados por gradiente (nombre de ejecución)",
    "GELU": "unidad lineal de error gaussiano (función de activación)",
    "GPU": "unidad de procesamiento gráfico",
    "GRU": "unidad recurrente con compuertas (celda de red neuronal recurrente)",
    "gru": "unidad recurrente con compuertas (nombre de ejecución)",
    "HRV": "variabilidad de la frecuencia cardiaca",
    "int8": "entero de 8 bits: pesos y activaciones cuantizados a enteros de 8 bits, como se ejecuta en el "
            "microcontrolador; _int8 = la versión int8 de una ejecución",
    "KB": "kilobyte (1024 bytes)",
    "log2": "logaritmo en base 2",
    "lr": "tasa de aprendizaje (tamaño de paso del optimizador)",
    "MAC": "operación de multiplicación-acumulación (una multiplicación más una suma): la unidad de coste de cómputo "
           "en el microcontrolador; MACs/latido = cuántas necesita el modelo por latido",
    "MACs": "operaciones de multiplicación-acumulación (ver MAC)",
    "MCU": "microcontrolador",
    "MHA": "atención multicabeza (capa de tipo transformer, no soportada por la cadena de herramientas del MCU)",
    "MLP": "perceptrón multicapa (red neuronal totalmente conectada)",
    "mlp": "perceptrón multicapa (nombre de ejecución)",
    "MMSE": "mínimo error cuadrático medio",
    "ms": "milisegundos",
    "MSE": "error cuadrático medio",
    "nn20": "número de diferencias sucesivas de RR mayores de 20 ms en la ventana",
    "nobp": "parte del nombre de ejecución: sin pasa banda (se quitan las características bp y bp_prev del pasa "
            "banda RSA fijo)",
    "np.corrcoef": "función de NumPy para el coeficiente de correlación de Pearson",
    "np.diff": "función de NumPy de diferencias sucesivas",
    "O(1)": "coste constante por latido (notación O grande); O(W) = proporcional a la longitud W de la ventana",
    "p": "valor p de la prueba de Wilcoxon pareada por sujeto: probabilidad de una diferencia al menos tan grande si "
         "los dos modelos fueran igual de precisos (p < 0.05 = significativo); en AR(p), el número de valores pasados",
    "B": "dimensión de lote (número de ventanas procesadas a la vez) en las formas de salida de las capas",
    "PSD": "densidad espectral de potencia (potencia de la señal por frecuencia)",
    "Q": "factor de calidad de un filtro pasa banda: frecuencia central / ancho de banda",
    "ReLU": "unidad lineal rectificada, max(0, x) (función de activación)",
    "RBJ": "fórmulas estándar de filtros biquad (segundo orden) de Robert Bristow-Johnson",
    "RLS": "mínimos cuadrados recursivos: filtro adaptativo que actualiza sus coeficientes en cada latido",
    "rls_ar": "referencia: pronóstico AR(8) adaptativo por paciente mediante RLS",
    "RMSE": "raíz del error cuadrático medio (ms): la medida principal de precisión, menor es mejor",
    "rmssd": "raíz cuadrada media de las diferencias sucesivas de los intervalos RR de la ventana",
    "RR": "intervalo RR: tiempo entre dos latidos consecutivos (ondas R del ECG), en ms",
    "rs": "rango reescalado: (máx - mín de las desviaciones acumuladas respecto a la media) / desviación estándar",
    "RSA": "arritmia sinusal respiratoria: la oscilación de la frecuencia cardiaca provocada por la respiración",
    "_rsa": "sufijo de nombre de ejecución: usa las características del seguidor de RSA",
    "SD1": "desviación estándar del diagrama de Poincaré perpendicular a la diagonal (variabilidad a corto plazo)",
    "SD2": "desviación estándar del diagrama de Poincaré a lo largo de la diagonal (variabilidad a largo plazo)",
    "seq": "nombre de conjunto de características / tipo de modelo: modelo de secuencia (toma la ventana RR cruda "
           "más características tabulares)",
    "SeparableConv1D": "convolución 1-D separable en profundidad: una convolución por canal seguida de una puntual "
                       "(mucho más barata que Conv1D)",
    "STM32H750": "microcontrolador de 32 bits de STMicroelectronics (Arm Cortex-M7, 128 KB de flash, 1 MB de RAM)",
    "RAM": "memoria de acceso aleatorio",
    "tab": "nombre de conjunto de características / tipo de modelo: modelo tabular (solo características diseñadas)",
    "TCN": "red convolucional temporal (convoluciones 1-D causales y dilatadas)",
    "tcn": "red convolucional temporal (parte del nombre de ejecución)",
    "TFLite": "TensorFlow Lite: entorno de ejecución y formato de archivo para modelos en dispositivos embebidos",
    "W": "longitud de la ventana: los 30 intervalos RR más recientes",
    "_ws": "sufijo de nombre de ejecución: arranque en caliente (inicializado con los pesos del padre, entradas "
           "nuevas a cero)",
    "z": "puntuación z: (valor - media) / desviación estándar, con estadísticas de los sujetos de entrenamiento",
    "ST": "STMicroelectronics (ST Edge AI: su cadena de herramientas para ejecutar redes neuronales en "
          "microcontroladores STM32)",
    "AI": "inteligencia artificial",
    "val": "validación: los 16 sujetos de validación, usados para seleccionar el modelo",
    "ar": "nombre de conjunto de características: entradas autorregresivas (rr_dev y los 8 dRR más recientes) del "
          "modelo lineal",
    "L2": "regularización L2: penalización sobre la suma de los cuadrados de los pesos del modelo",
    "float64": "números de coma flotante de 64 bits (doble precisión)",
    "nsrNNNRRcl": "código de sujeto de una serie RR de los registros de ritmo sinusal normal (NSR) (archivo "
                  "data/series/<código>.txt); los códigos puramente numéricos como 16273 o 006 son otros sujetos",
}
missing = [t for t, _, _ in G.GLOSSARY if t not in MEANING_ES]
assert not missing, f"tools/glossary_es.py: add Spanish meanings for {missing}"

# Spanish all-caps emphasis words used in the PDFs (not acronyms)
ALLOW_ES = {"CERRADO", "COMPLETA", "SELECCIONADO", "VERDADERO"}


def _ascii(text):
    # glossary.py's word boundaries are ASCII: without this the "p" of "pérdida" counts as the term p
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def used_terms(text):
    return [(t, MEANING_ES[t]) for t, _ in G.used_terms(_ascii(text))]


def undefined_tokens(text):
    return [t for t in G.undefined_tokens(_ascii(text)) if t not in ALLOW_ES]
