
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from .model import LoofylooPrime
from .data import get_data_loader

# This is a skeleton implementation of a training loop.
# To make this a functional training loop, you would need to
# implement a proper evaluation metric.

def main():
    """
    The main function for training the LoofylooPrime model.
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    training_config = config["training"]

    model = LoofylooPrime(
        text_encoder_name=model_config["text_encoder"],
        image_encoder_name=model_config["image_encoder"],
        audio_encoder_name=model_config["audio_encoder"],
        embed_dim=model_config["embed_dim"],
        num_experts=model_config["num_experts"],
        num_classes=model_config["num_classes"],
    )
    train_loader = get_data_loader(
        data_dir=training_config["data_dir"],
        batch_size=training_config["batch_size"],
        num_classes=model_config["num_classes"],
    )
    val_loader = get_data_loader(
        data_dir=training_config["val_data_dir"],
        batch_size=training_config["batch_size"],
        num_classes=model_config["num_classes"],
    )
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(training_config["num_epochs"]):
        # Training loop
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(
                batch["text"],
                batch["attention_mask"],
                batch["image"],
                batch["audio"],
            )
            target = batch["label"]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}/{training_config['num_epochs']}, Training Loss: {train_loss / len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                output = model(
                    batch["text"],
                    batch["attention_mask"],
                    batch["image"],
                    batch["audio"],
                )
                target = batch["label"]
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{training_config['num_epochs']}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
