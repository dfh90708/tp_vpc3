from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import torch ##
from torch.utils.data import DataLoader ##
import matplotlib.pyplot as plt
from logger_config import get_logger
import numpy as np
import tools as tools

class Trainer_Class:
    """
    Entrenar modelo.
    """

    def __init__(self, model, processor, data_train, data_valid, data_test, config):
        self.logger = get_logger()
        self.log_label = "[TRAIN]"
        self.trainer = None
        self.model = model
        self.processor = processor
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test #
        self.eval_dataloader = None
        self.config = config ##
        self.labels = None ##


    def __enter__(self):
        self.labels = self.data_train.features['label'].names ##
        self.train()
        self.eval() # 17-Junio - Sólo este no está probado.
        self.logger.info(f"{self.log_label} Entrenamiento completo.")
        return self

    def compute_metrics(self, eval_pred):
        log_sublabel = "[COMPUTE_METRICS]"
        try:
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)

            self.logger.info(f"{self.log_label} {log_sublabel} Compute metrics completo.")
            
            return {"accuracy": accuracy_score(labels, preds)}
            
        except Exception as e:
            self.logger.error(f"{self.log_label} {log_sublabel} Error: {e}")
            raise

    def train(self):
        log_sublabel = "[TRAIN]"
        try:
            training_args = TrainingArguments(
                output_dir= self.config['results'],  # 17-Junio     "./results",
                learning_rate=2e-5,
                per_device_train_batch_size=16,  
                per_device_eval_batch_size=16, 
                num_train_epochs=5,                 # 17-Junio se prueba con 1.
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                logging_steps=10,
                load_best_model_at_end=True
            )
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.data_train,
                eval_dataset=self.data_valid,
                tokenizer=self.processor,
                compute_metrics=self.compute_metrics
            )

            self.trainer.train()

        except Exception as e:
            self.logger.error(f"{self.log_label} {log_sublabel} Error: {e}")
            raise

    
    def collate_fn(self, batch):
        pixel_values = [item['pixel_values'] for item in batch]
        labels = [item['label'] for item in batch]
        return {
            'pixel_values': torch.stack(pixel_values),
            'labels': torch.tensor(labels)
        }
    
    def evaluate_model(self, model, dataloader, labels, dataset_raw, device='cpu'):
        model.eval()
        model.to(device)

        all_predictions = []
        all_labels = []
        
        # Para las primeras imágenes
        images_to_show = []
        predictions_to_show = []
        labels_to_show = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                pixel_values = batch['pixel_values'].to(device)
                labels_batch = batch['labels'].to(device)

                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

                if i < 1:
                    for j in range(min(15, len(labels_batch))):
                        images_to_show.append(dataset_raw[i * dataloader.batch_size + j]['image'])
                        predictions_to_show.append(predictions[j].item())
                        labels_to_show.append(labels_batch[j].item())

        # Convertimos a numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Detectar clases presentes en los datos
        present_class_indices = sorted(np.unique(np.concatenate([all_labels, all_predictions])))
        present_class_names = [labels[i] for i in present_class_indices]
        
        # Métricas
        acc = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, labels=present_class_indices, target_names=present_class_names))

        # Mostrar imágenes
        description = 'eval'
        actual_timestamp = tools.get_timestamp() 
        sample_path = f"{self.config['output_path']}/{actual_timestamp}_{description}_sample.png"
                    
        plt.figure(figsize=(20, 10))
        for i in range(len(images_to_show)):
            plt.subplot(3, 5, i + 1)
            plt.imshow(images_to_show[i])
            plt.title(f"Pred: {labels[predictions_to_show[i]]}\nTrue: {labels[labels_to_show[i]]}")
            plt.axis('off')
        plt.suptitle("Predicciones vs. Etiquetas verdaderas (primeras 15 imágenes)", fontsize=16)
        plt.tight_layout()
        plt.savefig(sample_path)
        #plt.show()


    def eval(self):
        log_sublabel = "[TRAIN]"
        try:
            log_sublabel = '[EVAL]'

            self.eval_dataloader = DataLoader(self.data_test, batch_size=32, shuffle=False, collate_fn=self.collate_fn)
            
            eval_results = self.trainer.evaluate()

            self.logger.info(f"{self.log_label}{log_sublabel} 'eval_results': {eval_results}")
            #print(eval_results)
            
            self.evaluate_model(
                model=self.model,
                dataloader=self.eval_dataloader,
                labels=self.labels,
                dataset_raw=self.data_test,     ##data_test_original,
                device='cuda' if torch.cuda.is_available() else 'cpu'
                )

        except Exception as e:
            self.logger.error(f"{self.log_label} {log_sublabel} Error: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(">>> 'exit' de class Trainer")  

