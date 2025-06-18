import yaml
from logger_config import get_logger
import tools as tools

from huggingface_hub import login

from datasets import load_dataset, Dataset, concatenate_datasets

import preprocess as prepro
import model as modelo
import trainer as train

from transformers import AutoImageProcessor


class Model_Pipeline:
    """
    Clase principal para encapsular todo el flujo:
    - Carga de configuración
    - Autenticación Hugging Face
    - Carga de datos
    - Preprocesamiento
    - Preparación del modelo
    - Entrenamiento
    """

    def __init__(self, config_path: str):
        self.logger = get_logger()
        self.logger.info(f"-------------------------------------------------------------------------------------------------")
        self.logger.info("Inicializando la pipeline ViT...")
        self.config = self.load_config(config_path)

        # Inicializar atributos
        self.dataset = None
        
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.data_test_original = None

        self.processor = None
        
        self.model_name = None
        self.model = None

        self.num_labels = None

        self.trainer = None

    def testing_components(self):
        log_label = "[TESTING]"
        self.logger.info(f"{log_label} Fecha_Hora de Tools: {tools.get_timestamp()}")

    def load_config(self, config_path: str) -> dict:
        log_label = "[CONFIGURACION]"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

                config_log = config.copy()
                if 'huggingface_token' in config_log:
                    config_log['huggingface_token'] = "***"                
                
                self.logger.info(f"{log_label} Configuración del modelo: {config_log}")
            return config
        except Exception as e:
            self.logger.error(f"{log_label} Error al cargar configuración: {e}")
            raise

    def authenticate(self):
        log_label = "[AUTENTICACION]"
        try:
            login(self.config["huggingface_token"])
            self.logger.info(f"{log_label} Hugging Face. Autenticación exitosa.")
        except Exception as e:
            self.logger.error(f"{log_label} Hugging Face. Error de autenticación: {e}")
            raise

    def load_data(self):
        log_label = "[CARGA DATASET]"
        try:
            self.logger.info(f"{log_label} Cargando dataset: '{self.config['dataset_name']}'")
            self.dataset = load_dataset(self.config["dataset_name"])
            self.logger.info(f"{log_label} \n '{self.dataset}'")
        except Exception as e:
            self.logger.error(f"{log_label} Error al intentar cargar el dataset: {e}")
            raise
    
    def preprocess_data(self):
        log_label = "[PREPROCESAR DATASET]"
        self.logger.info(f"{log_label} Inicializando preprocessor...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.config["model_name"])
            with prepro.Preprocess(self.dataset, self.config, self.processor) as ppr:
                self.data_train = ppr.data_train
                self.data_valid = ppr.data_valid
                self.data_test = ppr.data_test
                self.num_labels = ppr.num_labels
                self.data_test_original = ppr.data_test_original
            

            self.logger.info(f"{log_label} Finalizó preprocessor.")

        except Exception as e:
            self.logger.error(f"{log_label} Error: {e}")
            raise

    def prepare_model(self):
        log_label = "[PREPARE_MODEL]"
        with modelo.Model(self.config["model_name"], self.num_labels) as mdl:
            self.model = mdl.model

        self.logger.info(f"{log_label} Finalizó prepare_model.")
    
    def train_model(self):
        with train.Trainer_Class(self.model, self.processor, self.data_train, self.data_valid, self.data_test, self.data_test_original, self.config) as trn:
            self.trainer = trn.trainer

    def run(self):
        log_label = "[MAIN]"
        try:
            self.logger.info(f"{log_label} Ejecutando el pipeline...")
            
            self.testing_components()
            self.authenticate()
            self.load_data() 
            self.preprocess_data() 
            
            self.prepare_model()
            self.train_model()
            
            self.logger.info(f"{log_label} El Pipeline finalizó.")
        except Exception as e:
            self.logger.error(f"{log_label} Error en el Pipeline: {e}")
            raise

if __name__ == "__main__":
    # Punto de entrada principal
    pipeline = Model_Pipeline("config.yaml")
    pipeline.run()