import math
import torch
import torch.nn as nn
from enum import Enum
from torch.distributions.normal import Normal


class SpecialTokens(Enum):
    PADDING = 0
    SOS = 2
    EOS = 3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(p=dropout)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model
        )  # 1000^(2i/d_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/d_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/d_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]
        )


class ActorCritic(nn.Module):

    # Constructor
    def __init__(
        self,
        d_action: int,
        d_state: int,
        n_agents: int,
        d_model: int = 256,
        n_heads: int = 6,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
    ):
        super(ActorCritic, self).__init__()

        # INFO
        self.d_model = d_model

        # LAYERS
        self.log_action_std = nn.Parameter(torch.zeros(d_action, requires_grad=True))

        self.positional_encoder = PositionalEncoding(
            d_model=d_model, dropout=0.1, max_len=5000
        )

        self.special_token_embedding = nn.Embedding(
            len(SpecialTokens), d_model, padding_idx=0
        )

        self.state_embedding = nn.Linear(d_state, d_model, bias=False)
        self.action_embedding = nn.Linear(d_action, d_model, bias=False)

        # Encoder Params
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        # Decoder Params
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        self.out = nn.Linear(d_model, d_action)

        # Actor
        self.actor = (
            list(self.special_token_embedding.parameters())
            + list(self.state_embedding.parameters())
            + list(self.action_embedding.parameters())
            + list(self.positional_encoder.parameters())
            + list(self.enc.parameters())
            + list(self.dec.parameters())
            + list(self.out.parameters())
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(d_model * n_agents, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, deterministic=False, **kwargs):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        embedded_state = self.state_embedding(state)
        embedded_state = self.positional_encoder(embedded_state)

        encoder_out = self.enc(embedded_state)

        actions_buffer = kwargs.get("actions_buffer", [])

        if actions_buffer == []:
            tgt = encoder_out
        else:
            tgt = self.action_embedding(actions_buffer[-1])

        decoder_out = self.dec(tgt, memory=encoder_out)

        action_mean = self.out(decoder_out)

        if deterministic:
            return action_mean.detach()

        action_std = torch.exp(self.log_action_std)

        dist = Normal(action_mean, action_std)

        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        flattened_emb_state = torch.flatten(embedded_state, start_dim=1)
        state_val = self.critic(flattened_emb_state)

        return (
            action.detach(),
            action_logprob.detach(),
            state_val.detach(),
        )

    def evaluate(self, state, action):

        embedded_state = self.state_embedding(state.squeeze(1))
        embedded_state = self.positional_encoder(embedded_state)

        encoder_out = self.enc(embedded_state)

        decoder_out = self.dec(encoder_out, memory=encoder_out)

        action_mean = self.out(decoder_out)

        action_std = torch.exp(self.log_action_std)

        dist = Normal(action_mean, action_std)

        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)
        action_logprobs = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        flattened_emb_state = torch.flatten(embedded_state, start_dim=1)
        state_values = self.critic(flattened_emb_state)

        return action_logprobs, state_values, dist_entropy
