import torch 
import torch.nn as nn

# Visual based waypoint prediction 
class VisWaypointing(nn.Module):
    def __init__(self, latent_dim):
        super(VisWaypointing, self).__init__()

        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 3, 3)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 3, 3)
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (batch_size, 128*3*3)
        )

        # Fully connected layers for mean and log-variance
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

        # Fully connected layer to make predictions from latent z
        self.fc_pred_head = nn.Linear(latent_dim, 6 * 3)

    def encode(self, x):
        # Forward pass through encoder
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_logvar(encoded)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        bs = x.shape[0]
        # Forward pass through VAE
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        #print("z shape:", z.shape)
        #print("z flat shape:", z.flatten().shape)
        # preds shape after fc is (batchsize, 18) 
        pos_preds = self.fc_pred_head(z)
        #print("preds shape:", pos_preds.shape)
        # reshape to be consistent with original format (batchsize, 6, 3)
        pos_preds = pos_preds.reshape(bs, 6, 3)
        #print("preds re-shape:", pos_preds.shape)

        return mu, log_var, pos_preds
