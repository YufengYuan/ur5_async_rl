import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy


def weight_init(m):
    """Kaiming_normal is standard for relu networks, sometimes."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in",
            nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

OUT_DIM = {2: 18, 4: 7, 6: 1}
class EncoderModel(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, latent_dim, num_layers=4, num_filters=32):
        super().__init__()

        #assert len(obs_shape) == 3

        self.latent_dim = latent_dim
        self.num_layers = num_layers

        if type(obs_shape) is dict:
            self.obs_shape = obs_shape['observation']
            self.state_dim = obs_shape['proprioception'][0]
            self.init_conv(obs_shape[0], num_layers, num_filters)
            self.encoder_type = 'multimodal'
            self.latent_dim += self.state_dim
        elif len(obs_shape) == 3:
            self.obs_shape = obs_shape
            self.state_dim = 0
            self.init_conv(obs_shape[0], num_layers, num_filters)
            self.encoder_type = 'pixel'
            self.latent_dim = latent_dim
        elif len(obs_shape) == 1:
            self.obs_shape = None
            self.state_dim = obs_shape[0]
            self.encoder_type = 'state'
            self.latent_dim = obs_shape[0]


    def init_conv(self, in_channels, num_layers, num_filters):
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 2):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=2))
        self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)
        #self.outputs = dict()
        self.apply(weight_init)

    def forward_conv(self, obs):
        obs = obs / 255.
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        if self.encoder_type == 'state':
            return obs
        elif self.encoder_type == 'pixel':
            h = self.forward_conv(obs)
            if detach:
                h = h.detach()
            h_fc = self.fc(h)
            h_norm = self.ln(h_fc)
            out = torch.tanh(h_norm)
            return out
        elif self.encoder_type == 'multimodal':
            h = self.forward_conv(obs['observation'])
            if detach:
                h = h.detach()
            h_fc = self.fc(h)
            h_norm = self.ln(h_fc)
            out = torch.tanh(h_norm)
            out = torch.cat([out, obs['proprieception']], dim=-1)
            return out

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class ActorModel(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_dim, latent_dim=50, hidden_dim=1024, log_std_min=-10, log_std_max=2
    ):
        super().__init__()

        self.encoder = EncoderModel(obs_shape, latent_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent, action):
        latent_action = torch.cat([latent, action], dim=1)
        return self.trunk(latent_action)


class CriticModel(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_dim, latent_dim=50, hidden_dim=1024
    ):
        super().__init__()

        self.encoder = EncoderModel(obs_shape, latent_dim)

        self.Q1 = QFunction(
            self.encoder.latent_dim, action_dim, hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.latent_dim, action_dim, hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2



