# Comentarios
  17-Junio - Se .gitignore para omitir los archivos grandes.
  17-Junio - Está pendiente probar el eval() en trainer.py.
  16-Junio - Comentarios
                - En trainer.py, se modifica epochs a num_train_epochs=1 , solo para que la prueba sea corta. 
                          Valor original: num_train_epochs=5


# Metodología de desarrollo.
  - Se creó un entorno virtual.
  - Se creó un requirements.txt
  - Se diseñaron las clases involucradas con sus atributos y métodos.
  - Se implementaron las clases en componentes .py
  - Se utilizó "try/exception" para capurar los errores.
  - Se utilizó "logging" para registrar los mensajes en un único archivo log:
    - INFO: mensajes informativos para seguimiento de ejecución del pipeline.
    - ERROR: mensajes de eventos de error.
  - Se versionó con git.

  - Premisas consideradas:
    - Respetar los nombre de variables utilizado en el notebook MVP.
    - Crear clases para separar responsabilidades.
    - Escribir el mayor detalle posible en el archivo de log. 


# Ayuda

    # Instalar librerías
    pip install -r ./requirements.txt


    # Crear un Entorno Virtual
    python3 -m venv vpc3      # crear un entorno virtual
    source vpc3/bin/activate  # activar el entorno virtual
    deactivate                # desactivar el entorno virtual

    # Especificar carpetas para temporales del PIP. (en caso de fallas por falta de espacio en disco.)
    PIP_CACHE_DIR=/media/daniel/data/pip_cache TMPDIR=/media/daniel/data/temp pip install torchvision
    PIP_CACHE_DIR=/media/daniel/data/pip_cache TMPDIR=/media/daniel/data/temp pip install -U accelerate transformers



