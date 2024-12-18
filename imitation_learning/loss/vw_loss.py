import torch
import torch.nn as nn 

# Loss function
import torch
import torch.nn as nn

def vis_waypoint_loss(predicted_pos, true_pos, mu, log_var, beta=1.0):
    """
    Custom VAE loss function for predicting 3D positions.

    Args:
        predicted_pos: Tensor, predicted 3D positions (output from MLP), shape (batch_size, 3).
        true_pos: Tensor, ground truth 3D positions, shape (batch_size, 3).
        mu: Tensor, mean of the latent space distribution, shape (batch_size, latent_dim).
        log_var: Tensor, log variance of the latent space distribution, shape (batch_size, latent_dim).
        beta: Float, weight for the KL divergence term.

    Returns:
        loss: Total loss (scalar).
    """
    # Position loss (MSE between predicted and true positions)
    #print("predicted_pos shape:", predicted_pos.shape)
    #print("predicted_pos dtype:", predicted_pos.dtype)
    #print("predicted_pos type:", type(predicted_pos))
    #print("true_pos shape:", true_pos.shape)
    #print("true_pos dtype:", true_pos.dtype)
    #print("true_pos type:", type(true_pos))
    pos_loss = nn.MSELoss()(predicted_pos, true_pos)

    # KL divergence loss (encourages latent space to follow standard Gaussian)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss: Position loss + weighted KL divergence
    loss = pos_loss + beta * kl_loss

    return loss

