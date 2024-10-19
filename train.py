import albumentations as A
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    # outputs = model(torch.randn(16, 3, 32, 32))
    hf_dataset = load_dataset("uoft-cs/cifar10")

    train_dataset = DatasetWrapper(hf_dataset["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(10):
        for step, (inputs, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs = inputs.to("cuda")
            label = label.to("cuda")
            
            outputs = model(inputs)
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch} step: {step} loss: {loss.detach().item()}")