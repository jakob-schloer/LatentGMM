"""This file contains the vanilla VAE"""
import os
import torch
from torch import batch_norm, nn
import numpy as np
import pickle

import climvae.model.utils as ut

class BaseAE(nn.Module):
    """Auto encoder.

    Args:
        encoder ([type]): Encoder NN.
        decoder ([type]): Decoder NN.
    """

    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.encode = encoder
        self.decode = decoder

        # Loss functions
        self.track_loss = {
            'train': dict(loss=[]),
            'val': dict(loss=[])
        }

    def forward(self, x):
        """Pass through encoder and decoder.

        Args:
            x (tensor): (batch, x_dim)

        Return:
            x_hat (tensor): (batch, x_dim )
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, None, None

    def negative_elbo(self, x, l=None, save_loss=None):
        """Not actually a negative ELBO only the reconstruction loss.

        - L = - sum_z log(p(x|z))

        Args:
            x ([type]): [description]
            loss_save (str): Save loss of 'train' or 'val'. Default None.

        Returns:
            loss (torch.Tensor): Loss 
        """
        x_hat, z, _, _ = self.forward(x)
        # Reconstruction loss, log(p(x|z))
        # (batch)
        loss = ut.neg_log_likelihood(x, x_hat, dist='gaussian').mean()

        # track losses
        if save_loss is not None:
            try:
                self.track_loss[save_loss]['loss'].append(
                    loss.cpu().detach().numpy())
            except:
                print('Could not store loss of AE.')

        return loss
 


class AE1D(BaseAE):
    """Auto encoder with one dimensional latent space.

    Args:
        z_dim (int): Dimension of latent space. 
        encoder ([type]): Encoder NN.
        decoder ([type]): Decoder NN.
    """

    def __init__(self, encoder, decoder, z_dim=2, device=None):
        super().__init__(encoder, decoder, device=device)
        self.z_dim = z_dim

        # Loss functions
        self.track_loss = {
            'train': dict(loss=[]),
            'val': dict(loss=[])
        }
    
    def forward(self, x):
        """Pass through encoder and decoder.
        
        Args:
            x (tensor): (batch, x_dim)
        
        Return:
            x_hat (tensor): (batch, x_dim )
            z_given_x (tensor): (batch, z_dim)
        """
        z = self.encode(x)
        z = z.view(-1, 2, self.z_dim) # second layer
        q_m, q_logv = z[:, 0, :], z[:, 1, :] 
        z_given_x = ut.sample_gaussian(q_m, q_logv)
        x_hat = self.decode(z_given_x)

        return x_hat, z_given_x, q_m, q_logv
    

    def negative_elbo(self, x, l=None, save_loss=None):
        """Not actually a negative ELBO only the reconstruction loss.

        - L = - sum_z log(p(x|z))

        Args:
            x ([type]): [description]
            loss_save (str): Save loss of 'train' or 'val'. Default None.

        Returns:
            [type]: [description]

        TODO: Unfortune naming due to loss call in train.py. Change naming!
        """
        x_hat, z_given_x, q_m, q_logv = self.forward(x)
        # Reconstruction loss, log(p(x|z))
        # (batch)
        loss = ut.neg_log_likelihood(x, x_hat).mean()

        # track losses
        if save_loss is not None:
            try:
                self.track_loss[save_loss]['loss'].append(loss.cpu().detach().numpy())
            except:
                print('Could not store loss of AE.')

        return loss
    

class StackedAE(nn.Module):
    """Stacked auto encoder with one dimensional latent space."""

    def __init__(self, AEouter, AEinner, device=None):
        super().__init__()
        self.AEouter = AEouter 
        self.AEinner = AEinner 

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Loss functions
        self.track_loss = {
            'train': dict(loss=[]),
            'val': dict(loss=[])
        }
    

    def forward(self, x):
        """Pass through encoder and decoder.
        
        Args:
            x (tensor): (batch, x_dim)
        
        Return:
            x_hat (tensor): (batch, x_dim )
            z_given_x (tensor): (batch, z_dim)
        """
        z1 = self.AEouter.encode(x)

        z2 = self.AEinner.encode(z1)
        z2 = z2.view(-1, 2, self.AEinner.z_dim) 
        q_m, q_logv = z2[:, 0, :], z2[:, 1, :] 

        z2_given_x = ut.sample_gaussian(q_m, q_logv)
        z1_hat = self.AEinner.decode(z2_given_x)
        x_hat = self.AEouter.decode.forward(z1_hat)

        return x_hat, z2_given_x, q_m, q_logv
    

    def negative_elbo(self, x, l=None, save_loss=None):
        """Not actually a negative ELBO only the reconstruction loss.

        - L = - sum_z log(p(x|z))

        Args:
            x ([type]): [description]
            loss_save (str): Save loss of 'train' or 'val'. Default None.

        Returns:
            [type]: [description]

        """
        x_hat, z_given_x, q_m, q_logv = self.forward(x)
        # Reconstruction loss, log(p(x|z))
        # (batch)
        loss = ut.neg_log_likelihood(x, x_hat).mean()

        # track losses
        if save_loss is not None:
            try:
                self.track_loss[save_loss]['loss'].append(
                    loss.cpu().detach().numpy()
                )
            except:
                print('Could not store loss of AE.')

        return loss


class CNNAE(nn.Module):
    """Auto encoder.

    Args:
        z_dim (int): Dimension of latent space. 
        encoder ([type]): Encoder NN.
        decoder ([type]): Decoder NN.
    """

    def __init__(self, x_dim, z_dim, hid_channel=8):
        super().__init__()
        init_channel, dimx, dimy = x_dim
        self.z_dim = z_dim
     
        self. encoder = nn.Sequential(
            # 1st 
            nn.Conv2d(init_channel, hid_channel,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # 2nd 
            nn.Conv2d(hid_channel, hid_channel*2, 
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # 3rd 
            nn.Conv2d(hid_channel*2, hid_channel*4,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # 4th 
            nn.Conv2d(hid_channel*4, hid_channel*4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # 5th 
            nn.Conv2d(hid_channel*4, hid_channel*4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # 
            nn.Flatten(),
            nn.Linear((hid_channel*4)*(dimx//2**3)*(dimy//2**3), self.z_dim),            
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, (hid_channel*4)*(dimx//2**3)*(dimy//2**3)),
            nn.Unflatten(1, ((hid_channel*4), (dimx//2**3), (dimy//2**3))),
            nn.ReLU(),
            # 1st 
            nn.ConvTranspose2d(hid_channel*4, hid_channel*2, 
                               kernel_size=3, stride=2, padding=1, output_padding=1, 
                               bias=False),
            nn.ReLU(inplace=True),
            # 2nd
            nn.ConvTranspose2d(hid_channel*2, hid_channel,
                               kernel_size=3, stride=2, padding=1, output_padding=1, 
                               bias=False),
            nn.ReLU(inplace=True),
            # 3rd
            nn.ConvTranspose2d(hid_channel, init_channel,
                               kernel_size=3, stride=2, padding=1, output_padding=1, 
                               bias=False),
        )
        
    
    def forward(self, x):
        """Pass through encoder and decoder.
        
        Args:
            x (tensor): (batch, x_dim) Input to AE.
        
        Return:
            x_hat (tensor): (batch, x_dim ) Reconstruction of x.
            z (tensor): (batch, z_dim) Vector in latent space.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z 
    
    
    