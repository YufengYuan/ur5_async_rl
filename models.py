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
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 2):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=2))

        self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.out_dim = num_filters * out_dim * out_dim
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()
        self.apply(weight_init)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h


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
        self, encoder, action_dim, latent_dim=50, hidden_dim=1024, log_std_min=-10, log_std_max=2
    ):
        super().__init__()

        self.encoder = encoder

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.conv2latent = nn.Sequential(
            nn.Linear(encoder.out_dim, latent_dim),
            nn.LayerNorm(latent_dim), nn.Tanh(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        if detach_encoder:
            latent = self.encoder(obs).detach()
        else:
            latent = self.encoder(obs)
        latent = self.conv2latent(latent)
        #obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(latent).chunk(2, dim=-1)

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
        self, encoder, action_dim, latent_dim=50, hidden_dim=1024
    ):
        super().__init__()

        self.encoder = encoder

        self.conv2latent = nn.Sequential(
            nn.Linear(encoder.out_dim, latent_dim),
            nn.LayerNorm(latent_dim), nn.Tanh(),
        )
        self.Q1 = QFunction(
            latent_dim, action_dim, hidden_dim
        )
        self.Q2 = QFunction(
            latent_dim, action_dim, hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        if detach_encoder:
            latent = self.encoder(obs).detach()
        else:
            latent = self.encoder(obs)

        latent = self.conv2latent(latent)

        q1 = self.Q1(latent, action)
        q2 = self.Q2(latent, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2



