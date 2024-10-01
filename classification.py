import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import file_io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models



class ImageClassifierTrainer:
    def __init__(self, data_dir, batch_size=32, num_epochs=10, learning_rate=0.001, model_name='resnet18', model_path=None):
        """
        Initializes the ImageClassifierTrainer.

        :param data_dir: Directory where the dataset is located.
        :param batch_size: Batch size for training.
        :param num_epochs: Number of training epochs.
        :param learning_rate: Learning rate for the optimizer.
        :param model_name: Name of the pretrained model to use ('resnet18', 'resnet50', etc.).
        :param model_path: Path to a saved model to load. If None, a new model is initialized.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_data()
        self._build_model()
        if self.model_path:
            self.load_model(self.model_path)

    def _prepare_data(self):
        # Define data transformations with augmentation for training data
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # Mean for normalization
                                 [0.229, 0.224, 0.225])  # Std for normalization
        ])

        # Validation and test data transformations (no augmentation)
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # Load the dataset using the custom loader
        full_dataset = datasets.ImageFolder(self.data_dir, loader=file_io.rasterio_loader)

        # Split dataset into train, validation, and test sets
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42))

        # Apply transforms
        self.train_dataset.dataset.transform = self.train_transforms
        self.val_dataset.dataset.transform = self.test_transforms
        self.test_dataset.dataset.transform = self.test_transforms

        # Data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=0)

        self.class_names = full_dataset.classes

    def _build_model(self):
        # Define the model (using a pretrained model)
        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

        num_ftrs = self.model.fc.in_features
        num_classes = len(self.class_names)
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        best_val_loss = np.inf

        # Ensure the models directory exists
        models_dir = os.path.join('models', 'Classification')
        os.makedirs(models_dir, exist_ok=True)
        self.best_model_path = os.path.join(models_dir, 'best_model.pt')

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            train_loss = 0.0
            val_loss = 0.0

            # Training phase
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward and backward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)

            # Calculate average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            val_loss = val_loss / len(self.val_loader.dataset)

            print(f'Train Loss: {train_loss:.4f} \t Val Loss: {val_loss:.4f}')

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(
                    best_val_loss, val_loss))
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)

    def evaluate(self):
        # Load the best model
        self.load_model(self.best_model_path)

        # Evaluation on the test set
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.class_names)
        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, xticks_rotation='vertical')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self, path):
        """Saves the current model to the specified path."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Loads the model from the specified path."""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}. Starting with a new model.")

if __name__ == '__main__':
    data_dir = os.path.join('datasets', 'Classification')
    models_dir = os.path.join('models', 'Classification')
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, 'best_model.pt')

    # Initialize the trainer (first time, no model_path)
    trainer = ImageClassifierTrainer(data_dir=data_dir, batch_size=32, num_epochs=10, learning_rate=0.001)
    trainer.train()
    trainer.evaluate()

    # Reinitialize the trainer with the saved model
    trainer_loaded = ImageClassifierTrainer(data_dir=data_dir, model_path=best_model_path)
    trainer_loaded.evaluate()
