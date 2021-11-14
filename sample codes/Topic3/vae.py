import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

class Encoder(nn.Module):    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True
        
    def forward(self, x):
        x      = self.LeakyReLU(self.FC_input(x))
        x       = self.LeakyReLU(self.FC_input2(x))
        mean     = self.FC_mean(x)
        log_var  = self.FC_var(x)                   
                                                    
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x     = self.LeakyReLU(self.FC_hidden(x))
        x     = self.LeakyReLU(self.FC_hidden2(x))
        x_hat = torch.sigmoid(self.FC_output(x))
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)            
        z = mean + var*epsilon               
        return z
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

# Hyperparameters etc.
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 10
x_dim  = 784
hidden_dim = 400
latent_dim = 200
epochs = 5

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

writer_real = SummaryWriter(f"logs/vae/original")
writer_fake = SummaryWriter(f"logs/vae/output")
step = 0

from torch.optim import Adam

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

optimizer = Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(num_epochs):
    for batch_idx, (img, _) in enumerate(loader):
        img = img.view(batch_size, x_dim)
        img = img.to(device)

        x_hat, mean, log_var  = model(img)
        loss = loss_function(img, x_hat, mean, log_var)

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
                x_hat = x_hat.reshape(-1, 1, 28, 28)

                img_grid_original = torchvision.utils.make_grid(
                    img[:32], normalize=True
                )
                img_grid_output = torchvision.utils.make_grid(
                    x_hat[:32], normalize=True
                )

                writer_real.add_image("Original Image", img_grid_original, global_step=step)
                writer_fake.add_image("Output Image", img_grid_output, global_step=step)

            step += 1