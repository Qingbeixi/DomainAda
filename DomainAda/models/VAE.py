"created by qingjun, variational auto encoder"




import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

class VAE_1D(nn.Module):
    def __init__(self, input_dim=20, latent_dim=5, hidden_dim=10):
        super(VAE_1D, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.fc_mu = nn.Linear(latent_dim*12, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*12, latent_dim)

        # Decoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim*25)
        self.conv4 = nn.ConvTranspose1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(p=0.2)
        self.conv5 = nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)

        self.apply(weight_init)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # reshape from (batch_size, 50, 20) to (batch_size, 20, 50)
        x = self.bn1(F.relu(self.conv1(x))) # (batch_size, 10, 50)
        x = self.dropout1(self.pool1(x)) # (batch_size, 10, 25)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout2(self.pool2(x)) # (batch_size, 10, 12)
        x = self.bn3(self.conv3(x)) # (batch_size, 5, 12)
        x = x.view(x.size(0), -1) # (batch_size, 60)
        mu = self.fc_mu(x) # (batch_size, 5)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = F.relu(self.fc1(z)) # (batch_size, 5) -> (batch_size, 250)
        z = z.view(z.size(0), -1, 50)  # (batch_size, 5, 50)
        z = self.bn4(F.relu(self.conv4(z)))
        z = self.dropout4(z)
        x = torch.sigmoid(self.conv5(z))
        x = x.permute(0, 2, 1)  # reshape from (batch_size, 20, 50) to (batch_size, 50, 20)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

class VAERegressor(VAE_1D):
    def __init__(self, input_dim=20, latent_dim=5, hidden_dim=10):
        super(VAERegressor, self).__init__(input_dim, latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim*25, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def decode(self, z):
        h = F.relu(self.fc1(z)) # (batch_size, 250)
        y = self.sigmoid(self.fc4(h))
        return y
    
class VAE_rec(nn.Module):
    def __init__(self, input_dim=20, latent_dim=32, hidden_dim=10):
        super(VAE_rec, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.fc_mu = nn.Linear(latent_dim*12, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*12, latent_dim)

        # Decoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim*12)
        self.conv4 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.upsample1 = nn.Upsample(scale_factor=25/12, mode='nearest')
        self.dropout4 = nn.Dropout(p=0.2)
        self.conv5 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout5 = nn.Dropout(p=0.2)
        self.conv6 = nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)

        self.apply(weight_init)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # reshape from (batch_size, 50, 20) to (batch_size, 20, 50) 
        x = self.bn1(F.relu(self.conv1(x))) # (batch_size, 10, 50)
        x = self.dropout1(self.pool1(x)) # (batch_size, 10, 25)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout2(self.pool2(x)) # (batch_size, 10, 12)
        x = self.bn3(self.conv3(x)) # (batch_size, 5, 12)
        x = x.view(x.size(0), -1) # (batch_size, 60)
        mu = self.fc_mu(x) # (batch_size, 5)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = F.relu(self.fc1(z)) # (batch_size, 5) -> (batch_size, 120)
        z = z.view(z.size(0), -1, 12)  # (batch_size, 10, 12)
        z = self.bn4(F.relu(self.conv4(z))) # (batch_size, 10, 12)
        z = self.dropout4(self.upsample1(z)) # (batch_size, 10, 25)
        z = self.bn5(F.relu(self.conv5(z))) # (batch_size, 10, 25)
        z = self.dropout5(self.upsample2(z))  # (batch_size, 10, 50)
        z = F.relu(self.conv6(z)) # (batch_size, 20, 50)
        x = z.permute(0, 2, 1) # (batch_size, 20, 50)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    
