import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_polygon import OCRDataset, char2idx, idx2char
from model_cnn_transformer import OCRModel

# --- Hyperparameters ---
BATCH_SIZE = 32  # Giảm batch size cho ConvNeXt-Large
ACCUMULATION_STEPS = 2  # Gradient accumulation để tăng effective batch size
EPOCHS = 10
LEARNING_RATE = 5e-5  # Giảm learning rate cho ConvNeXt-Large
WARMUP_EPOCHS = 3  # Warmup epochs
WEIGHT_DECAY = 1e-4  # Weight decay cho AdamW
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = len(char2idx)
MODEL_SAVE_PATH = "ocr_model.pth"
BEST_MODEL_PATH = "best_ocr_model.pth"
SAVE_METRICS = True


def setup_environment():
    """Initialize environment and print setup information"""
    print(f"Using device: {DEVICE}")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Create a transforms object for data preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def collate_fn(batch):
    """Collate function for batching data"""
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets_input = [t[:-1] for t in targets]
    targets_output = [t[1:] for t in targets]
    targets_input = pad_sequence(
        targets_input, batch_first=True, padding_value=char2idx["<PAD>"]
    )
    targets_output = pad_sequence(
        targets_output, batch_first=True, padding_value=char2idx["<PAD>"]
    )
    return images, targets_input, targets_output


def load_datasets(transform):
    """Load and prepare datasets and dataloaders"""
    train_dataset = OCRDataset(
        image_dir=os.path.join("vietnamese", "train_images"),
        label_dir=os.path.join("vietnamese", "labels"),
        transform=transform,
    )

    test_dataset = OCRDataset(
        image_dir=os.path.join("vietnamese", "test_image"),
        label_dir=os.path.join("vietnamese", "labels"),
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset, train_loader, test_loader


def setup_model():
    """Initialize model, loss function, optimizer and scheduler"""
    model = OCRModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx["<PAD>"])

    # AdamW optimizer với weight decay cho ConvNeXt-Large
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # CosineAnnealingWarmRestarts với warmup cho ConvNeXt-Large
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Double the restart interval
        eta_min=1e-7,  # Minimum learning rate
    )

    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )

    return model, criterion, optimizer, scheduler, warmup_scheduler


def validate(model, dataloader, criterion):
    """Validate model on validation dataset"""
    model.eval()
    val_loss = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for images, tgt_input, tgt_output in dataloader:
            images, tgt_input, tgt_output = (
                images.to(DEVICE),
                tgt_input.to(DEVICE),
                tgt_output.to(DEVICE),
            )

            output = model(images, tgt_input)
            output = output.view(-1, VOCAB_SIZE)
            tgt_output = tgt_output.view(-1)

            loss = criterion(output, tgt_output)
            val_loss += loss.item()

            # Calculate character accuracy (ignore padding)
            pred = output.argmax(dim=1)
            mask = tgt_output != char2idx["<PAD>"]
            correct_chars += (pred[mask] == tgt_output[mask]).sum().item()
            total_chars += mask.sum().item()

    accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return val_loss / len(dataloader), accuracy


def train_epoch(model, train_loader, criterion, optimizer):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader)

    # Gradient accumulation
    optimizer.zero_grad()

    for batch_idx, (images, tgt_input, tgt_output) in enumerate(pbar):
        images, tgt_input, tgt_output = (
            images.to(DEVICE),
            tgt_input.to(DEVICE),
            tgt_output.to(DEVICE),
        )

        output = model(images, tgt_input)
        output = output.view(-1, VOCAB_SIZE)
        tgt_output = tgt_output.view(-1)

        loss = criterion(output, tgt_output)

        # Scale loss for gradient accumulation
        loss = loss / ACCUMULATION_STEPS
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights every ACCUMULATION_STEPS
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * ACCUMULATION_STEPS
        pbar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

    # Handle remaining gradients
    if len(train_loader) % ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    return epoch_loss / len(train_loader)


def plot_metrics(train_losses, val_losses, val_accuracies):
    """Plot and save training metrics"""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(val_accuracies) + 1),
        val_accuracies,
        label="Validation Accuracy",
        color="green",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Character Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
    plt.close()

    print("Training metrics visualization saved to 'training_metrics.png'")


def train_model():
    """Main training function"""
    # Setup
    transform = setup_environment()
    _, _, train_loader, test_loader = load_datasets(transform)
    model, criterion, optimizer, scheduler, warmup_scheduler = setup_model()

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("\n--- Starting Training ---\n")

    for epoch in range(EPOCHS):
        # Training
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss, val_accuracy = validate(model, test_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
        )

        # Save model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best model saved with val_loss: {val_loss:.4f}")

        # Update learning rate
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Print current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.2e}")

    # Plot and save metrics
    if SAVE_METRICS:
        plot_metrics(train_losses, val_losses, val_accuracies)

    print("\nTraining completed!")


if __name__ == "__main__":
    train_model()
