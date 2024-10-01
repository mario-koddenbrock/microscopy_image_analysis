import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.errors import RasterioIOError
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Ignore specific warnings
ignored_warnings = [
    rasterio.errors.NotGeoreferencedWarning,
    DeprecationWarning,
    FutureWarning,
    UserWarning
]

for warning in ignored_warnings:
    warnings.filterwarnings("ignore", category=warning)


class ImageClassifierTrainer:
    def __init__(self, data_dir, batch_size=32, num_epochs=10, learning_rate=0.001,
                 model_name='resnet18', model_path=None):
        """
        Initializes the ImageClassifierTrainer.

        :param data_dir: Directory where the dataset is located.
        :param batch_size: Batch size for training.
        :param num_epochs: Number of training epochs.
        :param learning_rate: Learning rate for the optimizer.
        :param model_name: Name of the pretrained model to use.
                            Available models: resnet18, resnet34, resnet50, resnet101, resnet152,
                            vgg11, vgg13, vgg16, vgg19,
                            densenet121, densenet169, densenet201, densenet161,
                            mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
                            efficientnet_b0 to efficientnet_b7, inception_v3
        :param model_path: Path to a saved model to load. If None, a new model is initialized.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name.lower()
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_data()
        self._build_model()
        if self.model_path:
            self.load_model(self.model_path)

        # For tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def _prepare_data(self):
        # Define default input size
        input_size = 224

        # Adjust input size for specific models
        if self.model_name == 'inception_v3':
            input_size = 299

        # Define data transformations with augmentation for training data
        self.train_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # Mean for normalization
                                 [0.229, 0.224, 0.225])  # Std for normalization
        ])

        # Validation and test data transformations (no augmentation)
        self.test_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # Define the custom image loader using Rasterio
        def rasterio_loader(path):
            try:
                with rasterio.open(path) as src:
                    image_array = src.read()  # Returns (bands, rows, cols)
                    # Handle band counts
                    if image_array.shape[0] == 1:
                        # Single band (grayscale), stack to create RGB
                        image_array = np.concatenate([image_array]*3, axis=0)
                    elif image_array.shape[0] > 3:
                        # More than 3 bands, take the first 3
                        image_array = image_array[:3]
                    # Transpose to (rows, cols, bands)
                    image_array = np.transpose(image_array, (1, 2, 0))
                    # Convert to uint8 if necessary
                    if image_array.dtype != np.uint8:
                        image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
                    # Convert to PIL Image
                    image = Image.fromarray(image_array)
                    return image
            except RasterioIOError as e:
                raise ValueError(f"Failed to load image {path}: {e}")

        # Load the dataset using the custom loader
        full_dataset = datasets.ImageFolder(self.data_dir, loader=rasterio_loader)

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

        # Data loaders with num_workers=0
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=0)

        self.class_names = full_dataset.classes

    def _build_model(self):
        num_classes = len(self.class_names)

        # Load the pretrained model
        self.model = self._get_pretrained_model(self.model_name, num_classes)

        self.model = self.model.to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_pretrained_model(self, model_name, num_classes):
        """Loads a pretrained model and modifies it for the current task."""
        model_name = model_name.lower()
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name.startswith('vgg'):
            model = getattr(models, model_name)(pretrained=True)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        elif model_name.startswith('densenet'):
            model = getattr(models, model_name)(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif model_name.startswith('mobilenet'):
            model = getattr(models, model_name)(pretrained=True)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        elif model_name.startswith('efficientnet'):
            model = getattr(models, model_name)(pretrained=True)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True)
            # Inception v3 has two classifiers
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            # Auxiliary classifier
            num_ftrs_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        return model

    def train(self):
        best_val_loss = np.inf

        # Ensure the models directory exists
        models_dir = os.path.join('models', 'Classification')
        os.makedirs(models_dir, exist_ok=True)
        self.model_path = os.path.join(models_dir, f'best_model_{self.model_name}.pt')

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            train_loss = 0.0
            val_loss = 0.0
            correct_train = 0
            total_train = 0
            correct_val = 0
            total_val = 0

            # Training phase
            self.model.train()
            train_loader = tqdm(self.train_loader, desc='Training', leave=False)
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward and backward pass
                if self.model_name == 'inception_v3':
                    # Inception v3 requires special handling
                    outputs, aux_outputs = self.model(inputs)
                    loss1 = self.criterion(outputs, labels)
                    loss2 = self.criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct_train += torch.sum(preds == labels.data)
                total_train += labels.size(0)

                # Update progress bar
                train_loader.set_postfix({'Loss': loss.item()})

            # Validation phase
            self.model.eval()
            val_loader = tqdm(self.val_loader, desc='Validation', leave=False)
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    if self.model_name == 'inception_v3':
                        outputs, _ = self.model(inputs)
                    else:
                        outputs = self.model(inputs)

                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)

                    # Calculate accuracy
                    _, preds = torch.max(outputs, 1)
                    correct_val += torch.sum(preds == labels.data)
                    total_val += labels.size(0)

                    # Update progress bar
                    val_loader.set_postfix({'Loss': loss.item()})

            # Calculate average losses and accuracies
            train_loss = train_loss / len(self.train_loader.dataset)
            val_loss = val_loss / len(self.val_loader.dataset)
            train_accuracy = correct_train.double() / len(self.train_loader.dataset)
            val_accuracy = correct_val.double() / len(self.val_loader.dataset)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy.item())
            self.val_accuracies.append(val_accuracy.item())

            print(f'Train Loss: {train_loss:.4f} \t Train Acc: {train_accuracy:.4f}')
            print(f'Val Loss: {val_loss:.4f} \t Val Acc: {val_accuracy:.4f}')

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                # print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(
                #     best_val_loss, val_loss))
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)

        # Optionally, plot the training and validation loss and accuracy curves
        self._plot_training_curves()


    def evaluate(self):
        # # Load the best model
        # self.load_model(self.model_path)

        # Evaluation on the test set
        self.model.eval()
        all_preds = []
        all_labels = []
        test_loader = tqdm(self.test_loader, desc='Testing', leave=False)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.model_name == 'inception_v3':
                    outputs, _ = self.model(inputs)
                else:
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

        # Print classification report
        self._print_classification_report(all_labels, all_preds)

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

    def _plot_training_curves(self):
        """Plots the training and validation loss and accuracy curves."""
        epochs = range(1, self.num_epochs + 1)

        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

        # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()


    def _print_classification_report(self, true_labels, predicted_labels):
        """
        Generates and prints a classification report.
        """
        report = classification_report(true_labels, predicted_labels, target_names=self.class_names)
        print("\nClassification Report:")
        print(report)


if __name__ == '__main__':
    data_dir = os.path.join('datasets', 'Classification')
    models_dir = os.path.join('models', 'Classification')
    os.makedirs(models_dir, exist_ok=True)

    # Specify the model architecture you want to use
    model_name = 'resnet50'  # Change this to the desired architecture

    # Initialize the trainer (first time, no model_path)
    trainer = ImageClassifierTrainer(
        data_dir=data_dir,
        batch_size=32,
        num_epochs=10,
        learning_rate=0.001,
        model_name=model_name
    )
    trainer.train()
    trainer.evaluate()

    # Reinitialize the trainer with the saved model
    trainer_loaded = ImageClassifierTrainer(
        data_dir=data_dir,
        model_name=model_name,
        model_path=os.path.join(models_dir, f'best_model_{model_name}.pt')
    )
    trainer_loaded.evaluate()
