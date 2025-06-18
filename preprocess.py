from transformers import AutoImageProcessor
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import Image

#import logging
import numpy as np
import matplotlib.pyplot as plt
import random
import tools as tools

from logger_config import get_logger

class Preprocess(object):
    """
    Preprocesamiento de los datos:
    - Dividisión del dataset en Train, Test y Valid.
    - Salvar imagen ejemplo al azar en 'outputs'
    - Informacion de los datasets.
    - Aplicar Transformaciones
    """

    def __init__(self, original_dataset, config, processor):
        self.logger = get_logger()
        self.original_dataset = original_dataset
        self.config = config
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.data_test_original = None
        self.processor = processor
        self.transform = None
        self.num_labels = None
    
    def __enter__(self):
        self.split_dataset()
        self.save_random_train_sample("pre-transform")
        self.get_information()
        self.get_transforms()
        self.preprocess_data()
        #self.save_random_train_sample("pos-transform")
        return self
    
    # [migrado]
    def save_random_train_sample(self, description):
        """
        Imprimir imagen ejemplo:
            pre-transform: imprime pre-transformaciones
            pos-transform: imprime pos-transformaciones
        """
        log_label = "[PREPROCESS SAVE TRAIN SAMPLE]"
        try:
            if description == 'pre-transform':
                self.logger.info(f"{log_label} Salvar imagen ejemplo de Train: {description}.")
                for i in range(self.config['samples_to_save']):
                    random_integer = random.randint(1, 300)
                    actual_timestamp = tools.get_timestamp() 
                    sample = self.data_train[random_integer]
                    sample_path = f"{self.config['output_path']}/{actual_timestamp}_{description}_train_sample_{random_integer}.png"
                    plt.imshow(sample['image'])
                    plt.title(f"label: {sample['label']}")
                    plt.axis("off")
                    plt.savefig(sample_path)  # Saves as PNG by default
                self.logger.info(f"{log_label} Imagenes salvadas. Ruta: {sample_path}")
                    
        except Exception as e:
            self.logger.error(f"{log_label} [{description}] Error: {e}")
            raise

    def get_information(self):
        log_label = "[PREPROCESS INFORMATION]"
        try:
            # Tamaño de los datasets
            self.logger.info(f"{log_label} Train size: {len(self.data_train)}")
            self.logger.info(f"{log_label} Test size: {len(self.data_test)}")
            self.logger.info(f"{log_label} Validation size: {len(self.data_valid)}")

            # Labels
            train_labels = self.data_train.features['label'].names
            test_labels = self.data_test.features['label'].names
            valid_labels = self.data_valid.features['label'].names
            # Nombre de Clases
            self.logger.info(f"{log_label} Train labels: {train_labels}")
            self.logger.info(f"{log_label} Test labels: {test_labels}")
            self.logger.info(f"{log_label} Validation labels: {valid_labels}")
            # Cantidad de Clases
            self.num_labels = len(train_labels)
            self.logger.info(f"{log_label} Train Class Quantity: {len(train_labels)}")
            self.logger.info(f"{log_label} Test Class Quantity: {len(test_labels)}")
            self.logger.info(f"{log_label} Validation Class Quantity: {len(valid_labels)}")

        except Exception as e:
            self.logger.error(f"{log_label} Error: {e}")
            raise  

    # [migrado]
    def split_dataset(self):
        log_label = "[PREPROCESS SPLIT]"
        try:
            dataset_dir = self.original_dataset
            split1 = dataset_dir['train'].train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
            temp_train = split1["train"]
            data_test = split1["test"]

            split2 = temp_train.train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
            data_train = split2["train"]
            data_valid = split2["test"]

            self.data_train = data_train
            self.data_valid = data_valid
            self.data_test = data_test
            self.data_test_original = data_test

            self.logger.info(f"{log_label} Crear dataset de Train, Valid y Test.")
        
        except Exception as e:
            self.logger.error(f"{log_label} Error: {e}")
            raise

    # [migrado]
    def get_transforms(self):
        log_label = "[GET TRANSFORMS]"
        try:
            self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.GaussianBlur(kernel_size=(3, 3)), 
            ])
            self.logger.info(f"{log_label} Configurar transformaciones: {str(self.transform)}")
            
            #return self.transform
        except Exception as e:
            self.logger.error(f"{log_label} Error: {e}")
            raise

    # [migrado]
    def transform_dataset(self, example, transform=None):
        log_label = "[TRANSFORM DATASET]"
        try:
            image = example['image']

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            if image.mode != "RGB":
                image = image.convert("RGB")

            if transform:
                image = transform(image)

            inputs = self.processor(images=image, return_tensors="pt")
            example['pixel_values'] = inputs['pixel_values'].squeeze(0)
            example['label'] = example['label']

            return example
        
        except Exception as e:
            self.logger.error(f"{log_label} Error: {e}")
            raise

    def preprocess_data(self):
        log_label = "[PREPROCESS DATA]"
        
        try:

            self.logger.info(f"{log_label} Transformacion a aplicar: {str(self.transform)}")

            self.data_train = self.data_train.map(lambda x: self.transform_dataset(x, self.transform), batched=False)
            self.data_valid = self.data_valid.map(lambda x:self.transform_dataset(x), batched=False)
            self.data_test = self.data_test.map(lambda x: self.transform_dataset(x), batched=False)

            self.data_train.set_format(type='torch', columns=['pixel_values','label'])
            self.data_valid.set_format(type='torch', columns=['pixel_values','label'])
            self.data_test.set_format(type='torch', columns=['pixel_values','label'])

            self.logger.info(f"{log_label} Pre-procesar datasets, aplicando transformaciones.")

        except Exception as e:
            self.logger.error(f"{log_label} Error: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(">>> 'exit' de class Proprocess")  
