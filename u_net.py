""" """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image

device = torch.device("cuda")

# encoder
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        pooled = self.pool(x)
        return x, pooled # x: skip connection, pooled: next encoder level

# decoder
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.layer1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x) # up convolution
        x = torch.cat([x, skip], dim=1)
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        return x

# unet class
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.conv3 = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # encoding
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        # bottleneck
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        # decoding
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        # final layer
        x = self.conv3(x)
        return x

# segmentation dataset
class PetSegDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask * 255).long().squeeze(0) - 1 # shifts 1,2,3 -> 0,1,2
        return image, mask

# evaluation function
def evaluate(model, test_loader):
    model.eval()
    TP = torch.zeros(3)
    FP = torch.zeros(3)
    FN = torch.zeros(3)
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            predictions = model(images).argmax(dim=1)
            for c in range(3):
                TP[c] += ((predictions == c) & (masks == c)).sum().item()
                FP[c] += ((predictions == c) & (masks != c)).sum().item()
                FN[c] += ((predictions != c) & (masks == c)).sum().item()
        IoU = TP / (TP + FP + FN + 1e-8)
        mIoU = IoU.mean()
        return mIoU

if __name__ == "__main__":
    # data
    train_data = OxfordIIITPet(root="./data", split="trainval", target_types="segmentation", download=True)
    test_data = OxfordIIITPet(root="./data", split="test", target_types="segmentation", download=True)
    # dataset
    train_set = PetSegDataset(train_data)
    test_set = PetSegDataset(test_data)
    # dataloader
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)

    model = UNet(num_classes=3).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(25):
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            # forward
            predictions = model(images)
            # loss
            loss = loss_fn(predictions, masks)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()

        print(f"Epoch {epoch}, loss = {loss.item():.4f}")
    
    # save to memory
    torch.save(model.state_dict(), "unet.pth")
    
    # evaluate
    miou = evaluate(model, test_loader)
    print(f"mIoU: {miou:.4f}")
    
    

            







        
