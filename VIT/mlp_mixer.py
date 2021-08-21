# implement the basic model of vision transformer
import jax.numpy as jnp
import torch
from torchvision import datasets, transforms
from torchsummary import summary
import torch.nn as nn
from einops.layers.torch import Rearrange
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MlpBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.):
        super(MlpBlock, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        x = self.net(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, input_dim, num_patch, token_dim, channel_dim, dropout=0.):
        super(MixerBlock, self).__init__()
        self.token_mixer = nn.Sequential(nn.LayerNorm(input_dim),
                                         Rearrange('b n d -> b d n'),
                                         MlpBlock(num_patch, token_dim, dropout=dropout),
                                         Rearrange('b d n -> b n d'))
        self.channel_mixer = nn.Sequential(nn.LayerNorm(input_dim),
                                           MlpBlock(input_dim, channel_dim, dropout))

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class MlpMixer(nn.Module):
    def __init__(self, in_channels, input_dim, num_classes, patch_size, image_size,
                 depth, token_dim, channel_dim, dropout=0.):
        super(MlpMixer, self).__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.to_embedding = nn.Sequential(nn.Conv2d(in_channels, input_dim, kernel_size=patch_size, stride=patch_size),
                                          Rearrange('b c h w -> b (h w) c'))
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(input_dim, self.num_patches, token_dim, channel_dim, dropout))
        self.layer_normal = nn.LayerNorm(input_dim)
        self.mlp_head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.to_embedding(x)
        for mixer in self.mixer_blocks:
            x = mixer(x)
        x = self.layer_normal(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x


mlp_mixer = MlpMixer(in_channels=3, input_dim=128, num_classes=10, patch_size=4, image_size=32,
                     depth=8, token_dim=128, channel_dim=256, dropout=0.5).to(device)
summary(mlp_mixer, (3, 32, 32))
train_dataset = datasets.CIFAR10(root="/home/liushuang/PycharmProjects/lab/mydata/CIFAR10",
                                 train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.CIFAR10(root="/home/liushuang/PycharmProjects/lab/mydata/CIFAR10",
                                train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp_mixer.parameters(), lr=0.01)


def train(epochs):
    mlp_mixer.train()
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            image, label = batch
            image, label = image.to(device), label.to(device)
            outputs = mlp_mixer(image)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 50 == 0:
                print("Epoch: {}, Step: {}, Loss: {}".format(epoch+1, i+1, loss.item()))


train(50)
torch.save(mlp_mixer.state_dict(), "/home/liushuang/PycharmProjects/lab/mymodel/mlp_mixer.pt")
mlp_mixer.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = mlp_mixer(images)
        _, predictions = torch.max(output, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum()

    accuracy = correct / total
    print("Accuracy on 10000 images is: {} %".format(100 * accuracy))
