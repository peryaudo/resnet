import albumentations as A
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ResNetModel

class DatasetWrapper(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset.with_format("numpy")
        self.transforms = A.Compose([
            A.Normalize()
        ])
    
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
    model = ResNetModel().to("cuda")
    loss_func = nn.CrossEntropyLoss()
    hf_dataset = load_dataset("uoft-cs/cifar10")

    train_dataset = DatasetWrapper(hf_dataset["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    val_dataset = DatasetWrapper(hf_dataset["test"])
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(10):
        for step, (inputs, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs = inputs.to("cuda")
            label = label.to("cuda")
            
            outputs = model(inputs)
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch} loss: {loss.detach().item()}")

        val_loss = 0.0
        correct_count = 0
        val_count = 0
        for step, (inputs, label) in enumerate(tqdm(val_dataloader)):
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