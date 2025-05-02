import math
import torch
import torch.nn as nn
from enum import Enum
from torch.distributions.normal import Normal


class SpecialTokens(Enum):
    PADDING = 0
    SOS = 2
    START_OF_STATE = 3
    START_OF_ACTION = 4
    EOS = 5


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
            torch.arange(0, d_model, 2).float() * (-math.log(1e5)) / d_model
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
        n_agents: int,
        d_state: int,
        d_action: int,
        device: str,
        # Model specific
        d_model: int = 64,
        n_heads: int = 1,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        train_max_agents: int = 16,
    ):
        super(ActorCritic, self).__init__()

        # INFO
        self.d_model = d_model
        self.n_agents = n_agents
        self.device = device
        self.d_action = d_action

        # LAYERS
        self.log_action_std = nn.Parameter(
            torch.zeros(d_action * train_max_agents, requires_grad=True, device=device)
        )

        self.positional_encoder = PositionalEncoding(
            d_model=d_model, dropout=0.1, max_len=500
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
        self.enc = nn.TransformerEncoder(
            encoder_layer, n_encoder_layers, enable_nested_tensor=False
        )

        # Decoder Params
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_action),
            nn.Tanh(),
        )

        # Actor
        self.actor_params = (
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
            nn.Linear(d_model, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self):
        raise NotImplementedError

    def get_value(self, state: torch.Tensor):
        with torch.no_grad():
            embedded_state = self.state_embedding(state)
            embedded_state = self.positional_encoder(embedded_state)
            encoder_out = self.enc(embedded_state)
            return self.critic(encoder_out[:, 0])

    def auto_regress(self, encoder_out):
        batch_dim = encoder_out.shape[0]
        tgt = self.special_token_embedding(
            torch.tensor(SpecialTokens.SOS.value, device=self.device)
        )
        action_means = []
        tgt = tgt.view(1, 1, self.d_model).repeat(batch_dim, 1, 1)
        for idx in range(self.n_agents):
            decoder_out = self.dec(
                tgt,
                memory=encoder_out,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    encoder_out.shape[1], device=self.device
                ),
                tgt_is_causal=True,
            )
            tgt = torch.cat([tgt, decoder_out[:, [-1], :]], dim=1)
            tgt = self.positional_encoder(tgt)
            action_means.append(self.out(decoder_out[:, [-1], :]))
        return torch.cat(action_means, dim=1)

    def act(self, state, deterministic=False, auto_regress=True):

        embedded_state = self.state_embedding(state)
        embedded_state = self.positional_encoder(embedded_state)

        encoder_out = self.enc(embedded_state)

        if auto_regress:
            action_mean = self.auto_regress(encoder_out)
        else:
            decoder_out = self.dec(
                tgt=encoder_out.clone().detach(),
                memory=encoder_out,
            )

            action_mean = self.out(decoder_out)

        if deterministic:
            return action_mean.flatten(start_dim=1).detach()

        action_std = torch.exp(
            self.log_action_std[: encoder_out.shape[1] * self.d_action]
        )

        dist = Normal(action_mean.flatten(start_dim=1), action_std)

        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        state_val = self.critic(encoder_out[:, 0])

        return (
            action.detach(),
            action_logprob.detach(),
            state_val.detach(),
        )

    def evaluate(self, state, action, causal=True):

        embedded_state = self.state_embedding(state)
        embedded_state = self.positional_encoder(embedded_state)

        encoder_out = self.enc(embedded_state)

        if causal:
            embedded_action = self.action_embedding(
                action.reshape(
                    action.shape[0],
                    self.n_agents,
                    self.d_action,
                )
            )
            embedded_action = self.positional_encoder(embedded_action)
            decoder_out = self.dec(
                tgt=embedded_action,
                memory=encoder_out,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    encoder_out.shape[1], device=self.device
                ),
                tgt_is_causal=True,
            )
        else:
            decoder_out = self.dec(encoder_out.clone().detach(), memory=encoder_out)

        action_mean = self.out(decoder_out)

        action_std = torch.exp(
            self.log_action_std[: encoder_out.shape[1] * self.d_action]
        )

        dist = Normal(action_mean.flatten(start_dim=1), action_std)

        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        state_values = self.critic(
            encoder_out[:, 0].detach()
        )  # Detach is important otherwise i get backprop error

        return action_logprob, state_values, dist_entropy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActorCritic(
        n_agents=4,
        d_state=18,
        d_action=2 * 4,
        device=device,
    ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
