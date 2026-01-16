# Unified MD Analysis

Software unificado para análisis de estructuras cristalinas FCC/BCC en simulaciones de dinámica molecular LAMMPS.

## 📋 Características

- **Alpha Shape**: Detección de superficie con Ghost Particles
- **Clustering**: Separación de nanoporos con HDBSCAN, KMeans, MeanShift, Agglomerative
- **Preprocesamiento**: Extracción de 37 features geométricas para Machine Learning
- **Training**: Entrenamiento de Random Forest para predicción de vacancias
- **Predicción**: Inferencia de vacancias en nuevos dumps

## 🏗️ Estructura del Proyecto

```
unified_md_analysis/
├── core/
│   ├── surface_detection.py  # Alpha Shape + Ghost Particles
│   ├── clustering.py          # Clustering (HDBSCAN, KMeans, etc.)
│   ├── preprocessing.py       # Extracción de 37 features
│   ├── training.py            # Entrenamiento Random Forest
│   └── prediction.py          # Predicción de vacancias
├── utils/
│   ├── constants.py           # Constantes globales
│   └── lammps_parser.py       # Parser LAMMPS unificado
├── cli/
│   ├── alpha_shape.py         # CLI detección de superficie
│   ├── cluster.py             # CLI clustering
│   ├── preprocess.py          # CLI extracción de features
│   ├── train.py               # CLI training
│   └── predict.py             # CLI predicción
├── main.py                    # Orquestador principal
└── requirements.txt
```

## 🚀 Instalación

```bash
# Clonar o copiar el directorio
cd unified_md_analysis

# Instalar dependencias
pip install -r requirements.txt

# Opcional: HDBSCAN (clustering avanzado)
pip install hdbscan
```

## 📖 Uso

### 1️⃣ Alpha Shape (Detección de Superficie)

Detecta átomos superficiales eliminando bulk:

```bash
python main.py alpha_shape input.dump output_surface.dump

# Con parámetros personalizados
python main.py alpha_shape input.dump output.dump \
    --probe-radius 2.2 \
    --num-ghost-layers 3 \
    --smoothing 10
```

**Parámetros:**
- `--probe-radius`: Radio de sonda en Å (default: 2.0)
- `--lattice-param`: Parámetro de red (default: auto-detectar)
- `--num-ghost-layers`: Capas fantasma para PBC (default: 2)
- `--smoothing`: Iteraciones de suavizado (default: 0)

---

### 2️⃣ Clustering (Opcional)

Separa nanoporos individuales:

```bash
# HDBSCAN (automático)
python main.py cluster surface.dump clusters_dir/ --method hdbscan

# KMeans (manual)
python main.py cluster surface.dump clusters_dir/ --method kmeans --n-clusters 5

# MeanShift (automático)
python main.py cluster surface.dump clusters_dir/ --method meanshift
```

**Salida:** Directorio con `cluster_0.dump`, `cluster_1.dump`, etc.

---

### 3️⃣ Preprocesamiento (Extracción de Features)

Extrae 37 features geométricas para Machine Learning:

```bash
python main.py preprocess surface_dumps_dir/ --output features.csv

# Con vacancias conocidas (para training)
python main.py preprocess dumps/ --output features.csv --vacancies-file vacancies.txt
```

**Features extraídas (37 total):**
- 26 del grid 3D (ocupación, gradientes, fragmentación, etc.)
- 2 del Convex Hull (volumen, área)
- 3 momentos de inercia principales
- 2 radiales (RDF mean, kurtosis)
- 1 entropía espacial
- 1 bandwidth de clustering

---

### 4️⃣ Training

Entrena modelo Random Forest:

```bash
python main.py train features.csv --output models/

# Con parámetros personalizados
python main.py train features.csv --output models/ \
    --n-estimators 200 \
    --test-size 0.3
```

**Salida:**
- `modelo_rf.joblib`: Modelo entrenado
- `feature_names.txt`: Nombres de features
- `feature_importance.csv`: Importancias

---

### 5️⃣ Predicción

Predice vacancias en nuevos dumps:

```bash
# Un archivo
python main.py predict models/modelo_rf.joblib new_dump.dump

# Múltiples archivos
python main.py predict models/modelo_rf.joblib dumps_dir/ --output predictions.csv
```

---

## 🔧 Pipeline Completo (Ejemplo)

```bash
# 1. Detectar superficie (Alpha Shape)
for dump in raw_dumps/*.dump; do
    python main.py alpha_shape "$dump" "surface_dumps/$(basename $dump)"
done

# 2. Preprocesar: Extraer features (con vacancias conocidas)
python main.py preprocess surface_dumps/ --output features.csv --vacancies-file vacancies.txt

# 3. Entrenar modelo
python main.py train features.csv --output models/

# 4. Predecir en nuevos dumps
python main.py predict models/modelo_rf.joblib new_dumps/ --output predictions.csv
```

---

## 📊 Constantes Clave

Definidas en `utils/constants.py`:

- **A0**: 3.532 Å (parámetro de red FCC Cu)
- **ATM_TOTAL**: 16384 (átomos totales esperados)
- **GRID_SIZE**: 10×10×10 (grid de ocupación)
- **DEFAULT_PROBE_RADIUS**: 2.0 Å

---

## 🧩 Algoritmos de Clustering

| Algoritmo | Descripción | Cuándo usar |
|-----------|-------------|-------------|
| **HDBSCAN** | Jerárquico basado en densidad | Automático, detecta ruido |
| **KMeans** | Particionamiento en K clusters | Número conocido de clusters |
| **MeanShift** | Basado en densidad | Estimación automática |
| **Agglomerative** | Jerárquico aglomerativo | Dendrogramas, linkage |

---

## 📦 Dependencias

**Core:**
- numpy
- pandas
- scipy
- scikit-learn
- joblib

**Opcional:**
- hdbscan (clustering avanzado)
- ovito (extracción de superficie con OVITO)

---

## 🔬 Código Recomendado por Etapa

Basado en el análisis del repositorio original:

| Etapa | Código Base | Razón |
|-------|-------------|-------|
| **Alpha Shape** | `alpha_shape_ghost_particles.py` | Clase modular, sin Streamlit, auto-detecta lattice |
| **Clustering** | `cluster_app_spirit.py` | 4 algoritmos, métricas completas |
| **Preprocesamiento** | `simplified_extractor_enhanced.py` | 37 features, PCA optimizado |
| **Training** | `train_simplified.py` | Código limpio, 330 líneas |
| **Predicción** | `vacancy_batch_predict.py` | Batch optimizado, consistente |

---

## 📝 Ventajas del Software Unificado

✅ **Sin Streamlit**: CLI puro, sin dependencias de UI
✅ **Modular**: Cada etapa es independiente y reutilizable
✅ **Consistente**: Parser LAMMPS y constantes unificadas
✅ **Documentado**: Docstrings completas en cada módulo
✅ **Extensible**: Fácil agregar nuevos algoritmos

---

## 🎯 Nomenclatura Correcta

- **Alpha Shape** = Detección de superficie (NO es preprocesamiento)
- **Preprocesamiento** = Extracción de features (preparación para ML)
- **Training** = Entrenamiento del modelo
- **Predicción** = Inferencia de vacancias

---

## 🤝 Contribuciones

Para agregar nuevos algoritmos o features:

1. Edita los módulos en `core/`
2. Actualiza `FEATURE_ORDER` en `utils/constants.py` si cambias features
3. Crea tests para validar consistencia

---

## 📄 Licencia

[Especificar licencia del proyecto]

---

## 📧 Contacto

[Información de contacto]
