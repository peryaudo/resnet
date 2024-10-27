import hydra
from omegaconf import OmegaConf
import albumentations as A
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import wandb

from model import resnet18, resnet34, resnet50

class DatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transforms=A.Normalize(), image_key="img", label_key="label"):
        self.image_key = image_key
        self.label_key = label_key
        self.hf_dataset = hf_dataset.with_format("numpy")
        self.transforms = transforms
    
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx][self.image_key]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
        image = image[:,:,0:3]
        image = self.transforms(image=image)["image"]
        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1))
        label = torch.tensor(int(self.hf_dataset[idx][self.label_key]))
        return image, label

@hydra.main(version_base=None, config_path=".", config_name="config_cifar10")
def train_main(cfg):
    if cfg["dataset"] == "cifar10":
        hf_dataset = load_dataset("uoft-cs/cifar10")
        train_dataset = DatasetWrapper(hf_dataset["train"], A.Compose([
            A.CropAndPad(px=4, p=1.0, keep_size=False),
            A.RandomCrop(height=32, width=32, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
        ]))
        val_dataset = DatasetWrapper(hf_dataset["test"])
        num_classes = 10
    elif cfg["dataset"] == "imagenet-1k":
        hf_dataset = load_dataset("ILSVRC/imagenet-1k", token=True, trust_remote_code=True)
        train_dataset = DatasetWrapper(hf_dataset["train"], A.Compose([
            A.PadIfNeeded(min_height=224, min_width=224),
            A.RandomCrop(height=224, width=224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
        ]), image_key="image")
        val_dataset = DatasetWrapper(hf_dataset["validation"], A.Compose([
                A.PadIfNeeded(min_height=224, min_width=224),
                A.CenterCrop(height=224, width=224, p=1.0),
                A.Normalize(),
            ]), image_key="image")
        num_classes = 1000
    else:
        raise ValueError(f"dataset type {cfg['dataset']} not supported")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=32)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=32)

    if cfg["model"] == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif cfg["model"] == "resnet34":
        model = resnet34(num_classes=num_classes)
    elif cfg["model"] == "resnet50":
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"model type {cfg['model']} not supported")

    if cfg["dataset"] == "cifar10":
        model.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    model = model.to("cuda")
    model = torch.compile(model)
    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=len(train_dataloader))

    # TODO: Match the performance shown in https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html

    wandb.init(project="resnet", name=cfg.get("run_name", None), config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    global_steps = 0
    for epoch in range(cfg["epochs"]):
        for inputs, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            inputs = inputs.to("cuda", non_blocking=True)
            label = label.to("cuda", non_blocking=True)
            
            outputs = model(inputs)
            loss = loss_func(outputs, label)
            loss.backward()
            if global_steps % 100 == 0:
                wandb.log({"train/loss": loss.detach().item(), "train/leraning_rate": lr_scheduler.get_last_lr()[0]}, step=global_steps)
            optimizer.step()
            lr_scheduler.step()
            global_steps += 1
        print(f"epoch: {epoch} loss: {loss.detach().item()}")

        val_loss = 0.0
        correct_count = 0
        val_count = 0
        for inputs, label in tqdm(val_dataloader):
            with torch.no_grad():
                inputs = inputs.to("cuda", non_blocking=True)
                label = label.to("cuda", non_blocking=True)
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

if __name__ == "__main__":
    train_main()