import argparse

import albumentations as A
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from basic_model import resnet18

class DatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transforms=A.Normalize()):
        self.hf_dataset = hf_dataset.with_format("numpy")
        self.transforms = transforms
    
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]["img"]
        image = self.transforms(image=image)["image"]
        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1))
        label = torch.tensor(int(self.hf_dataset[idx]["label"]))
        return image, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Name of the run")
    args = parser.parse_args()

    wandb.init(project="resnet", name=args.run_name)

    model = resnet18().to("cuda")
    loss_func = nn.CrossEntropyLoss()
    hf_dataset = load_dataset("uoft-cs/cifar10")

    train_dataset = DatasetWrapper(hf_dataset["train"], A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.5),
        A.Cutout(num_holes=3,
                 max_h_size=6,
                 max_w_size=6,
                 fill_value=0,
                 p=0.5),
        A.Normalize(),
    ]))
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
    val_dataset = DatasetWrapper(hf_dataset["test"])
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    global_steps = 0

    # TODO: Match the performance shown in https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html

    for epoch in range(20):
        for inputs, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            inputs = inputs.to("cuda")
            label = label.to("cuda")
            
            outputs = model(inputs)
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()
            if global_steps % 100 == 0:
                wandb.log({"train/loss": loss.detach().item()}, step=global_steps)
            global_steps += 1
        print(f"epoch: {epoch} loss: {loss.detach().item()}")

        val_loss = 0.0
        correct_count = 0
        val_count = 0
        for inputs, label in tqdm(val_dataloader):
            with torch.no_grad():
                inputs = inputs.to("cuda")
                label = label.to("cuda")
                outputs = model(inputs)
                loss = loss_func(outputs, label)
                val_loss += loss.detach().item() * inputs.size(0)
                val_count += inputs.size(0)

                pred = torch.argmax(outputs, dim=-1)

                correct_count += torch.sum((pred == label).long()).detach().item()

        accuracy = correct_count / val_count
        val_loss /= val_count
        print(f"accuracy: {accuracy} val_loss: {val_loss}")
        wandb.log({"val/loss": val_loss, "val/accuracy": accuracy}, step=global_steps)
    wandb.finish()