import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 28 * 28), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 10

model = autoencoder().cuda()

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

writer_real = SummaryWriter(f"logs/autoenc/original")
writer_fake = SummaryWriter(f"logs/autoenc/output")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (img, _) in enumerate(loader):
        img = img.view(-1, 784).to(device)
        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                     Loss: {loss:.4f}"
            )

            with torch.no_grad():
                # take out (up to) 32 examples
                img = img.reshape(-1, 1, 28, 28)
                output = output.reshape(-1, 1, 28, 28)

                img_grid_original = torchvision.utils.make_grid(
                    img[:32], normalize=True
                )
                img_grid_output = torchvision.utils.make_grid(
                    output[:32], normalize=True
                )

                writer_real.add_image("Original Image", img_grid_original, global_step=step)
                writer_fake.add_image("Output Image", img_grid_output, global_step=step)

            step += 1