from transformers import AutoModelForImageClassification
from logger_config import get_logger

#logger = logging.getLogger(__name__)

class Model:
    """
    Crear modelo.
    """

    def __init__(self, model_name, num_labels):
        self.logger = get_logger()
        self.log_label = "[MODEL]"
        self.model = None
        self.model_name = model_name
        self.num_labels = num_labels

    def __enter__(self):
        self.logger.info(f"{self.log_label} Valores - model_name = {self.model_name} , num_labels = {self.num_labels} .")
        self.get_model()
        return self

    def get_model(self):   
        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True
            )
            self.logger.info(f"{self.log_label} Modelo {self.model_name} creado con {self.num_labels}")

        except Exception as e:
            self.logger.error(f"{self.log_label} Error: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(">>> 'exit' de class Model")  
