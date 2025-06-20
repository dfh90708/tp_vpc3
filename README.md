# 🧠 ViT Image Classification Pipeline

Este proyecto implementa un pipeline completo para clasificación de imágenes utilizando modelos Vision Transformer (ViT). Está diseñado para entrenar, evaluar y registrar modelos con trazabilidad y modularidad, incorporando MLflow para experimentación reproducible.

---

## Estructura del Proyecto

```
.
└── vpc3/
    └── (entorno virtual)
├── .gitignore
├── requirements.txt
├── requirements_torch.txt
├── config.yaml
├── logger_config.py
├── tools.py
├── main.py
├── preprocess.py
├── model.py
├── trainer.py
├── logs/
│   └── pipeline.log
├── output/
│   └── (imagenes ejemplo pre-transform)
├── results/
│   └── (checkpoints del modelo)
├── mlartifacts/
├── mlruns/
```

---

## Metodología de Desarrollo

- Creación de entorno virtual (`vpc3`)
- Definición de `requirements.txt` y `requirements_torch.txt` para dependencias
- Modularización del código en clases: `Model_Pipeline`, `Preprocess`, `Model`, `Trainer`
- Manejo de errores con `try/except`
- Uso de `logging` para trazabilidad en consola y archivo (`logs/pipeline.log`)
  - INFO: mensajes informativos para seguimiento de ejecución del pipeline.
  - ERROR: mensajes de eventos de error.
- Registro y versionado con `git`
- Registro de experimentos con MLflow

---

## Configuración

El archivo `config.yaml` centraliza todos los parámetros del pipeline:

```yaml
huggingface_token: "xxxxxxxxxxxxxxxxxxxxxxx"
output_path: "./output"
samples_to_save: 3
results: "./results"
dataset_name: "gymprathap/Breast-Cancer-Ultrasound-Images-Dataset"
model_name: "facebook/deit-base-patch16-224"
batch_size: 16
num_epochs: 5
learning_rate: 2e-5
tracking_url: http://localhost:5000
```

---

## Ejecución

### 1. Crear entorno virtual

```bash
python3 -m venv vpc3
source vpc3/bin/activate       # En Windows (Git Bash): source vpc3/Scripts/activate
```

### 2. Instalar dependencias

Con GPU:

```bash
pip install -r requirements.txt
pip install -r requirements_torch.txt --index-url https://download.pytorch.org/whl/cu124
```

Sin GPU:

```bash
pip install -r requirements.txt
pip install -r requirements_torch.txt
```

> Tip para espacio limitado:

```bash
PIP_CACHE_DIR=/media/data/pip_cache TMPDIR=/media/data/temp pip install torchvision
```

### 3. Iniciar MLflow

Asegurate de tener un servidor MLflow corriendo localmente:

```bash
mlflow ui
```

Abrirá una interfaz en: [http://localhost:5000](http://localhost:5000)

Los experimentos se guardan en:

- `mlruns/`: metadatos de ejecuciones
- `mlartifacts/`: artefactos (modelos, imágenes, etc.)

---

### 4. Ejecutar el pipeline

```bash
python main.py
```

---

## Descripción de Componentes

### `main.py`

Orquesta todo el flujo:

- Carga la configuración
- Autentica en Hugging Face
- Carga y preprocesa datos
- Prepara modelo
- Entrena y evalúa

### `preprocess.py`

- Divide el dataset en `train`, `validation`, `test`
- Aplica transformaciones de data augmentation
- Convierte imágenes con `AutoImageProcessor`
- Guarda imágenes ejemplo (pre-transformación) en `output/`

### `model.py`

Carga el modelo ViT desde `transformers` con el número de clases detectado.

### `trainer.py`

- Entrena con `Trainer` de Hugging Face
- Evalúa con métricas (`accuracy`, `f1`, `precision`, `recall`)
- Loguea métricas y artefactos en MLflow
- Guarda predicciones con imágenes en `output/`

---

## Estado del Proyecto

- Preprocesamiento y entrenamiento funcional
- Logging detallado implementado
- Evaluación con visualización de resultados
- Evaluación (`eval()` en `trainer.py`) pendiente de validación completa

---

## Notas Finales

- Las transformaciones en `preprocess.py` usan `torchvision.transforms` con rotación, zoom, blur, y crop.
- Se usan `context managers` (`__enter__`, `__exit__`) para asegurar inicialización/limpieza adecuada en clases como `Model`, `Preprocess`, y `Trainer_Class`.
- Se loguean muestras de imágenes, métricas y modelos con MLflow para trazabilidad completa.
- Todo el pipeline puede reconfigurarse fácilmente desde `config.yaml`.
