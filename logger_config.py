import logging
import os

def get_logger(log_file: str = "logs/pipeline.log") -> logging.Logger:
    # Crear carpeta logs si no existe
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Crear logger
    logger = logging.getLogger("ViT_Pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Evita duplicados si se llama varias veces

    # Evita Handlers duplicados
    if not logger.handlers:
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)

        # Handler para archivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)

        # Agregar ambos handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
