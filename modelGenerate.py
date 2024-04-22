import math
import torch
import torch.nn as nn

from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RWKV_TimeMix_x051a(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd

            self.time_maa_k = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * \
                    (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd, eps=(1e-5)*64)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        H, N = self.n_head, self.head_size
        if T % 256 == 0:
            Q = 256
        elif T % 128 == 0:
            Q = 128
        else:
            Q = T
        assert T % Q == 0

        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2)  # receptance
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1)  # key
        v = self.value(xv).view(B, T, H, N).transpose(1, 2)  # value
        g = F.silu(self.gate(xg))  # extra gate

        w = torch.exp(-torch.exp(self.time_decay.float()))  # time_decay
        u = self.time_faaaa.float()  # time_first

        ws = w.pow(Q).view(1, H, 1, 1)

        ind = torch.arange(
            Q-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Q).pow(ind)

        wk = w.view(1, H, 1, Q)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Q))
        w = torch.tile(w, [Q])
        w = w[:, :-Q].view(-1, Q, 2*Q - 1)
        w = w[:, :, Q-1:].view(1, H, Q, Q)

        w = w.to(dtype=r.dtype)  # the decay matrix
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        state = torch.zeros(B, H, N, N, device=r.device,
                            dtype=r.dtype)  # state
        y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype)  # output

        for i in range(T // Q):  # the rwkv-x051a operator
            rr = r[:, :, i*Q:i*Q+Q, :]
            kk = k[:, :, :, i*Q:i*Q+Q]
            vv = v[:, :, i*Q:i*Q+Q, :]
            y[:, :, i*Q:i*Q+Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            state = ws * state + (kk * wk) @ vv

        y = y.transpose(1, 2).contiguous().view(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g

        # output projection
        y = self.dropout(self.output(y))
        return y


class RWKV_ChannelMix_x051a(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.time_maa_k = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(config.n_embd, 3 *
                             config.n_embd, bias=config.bias)
        self.value = nn.Linear(
            3 * config.n_embd, config.n_embd, bias=config.bias)
        self.receptance = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x
        x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps)


class GroupedQAttention(nn.Module):
    def __init__(self, dim, num_heads, groups=4):
        super().__init__()
        self.num_heads = num_heads
        self.groups = groups

        self.qkvw = nn.Linear(dim, dim * 4, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        qkvw = self.qkvw(x)  # GENERATE
        qkvw_gropus = torch.chunk(qkvw, self.groups, dim=-1)  # GENERATE
        q, k, v, w = [t.chunk(self.groups, dim=-1) for t in qkvw_gropus]

        q, k, v, w = [
            torch.cat([qi, ki, vi, wi], dim=0)
            for qi, ki, vi, wi in zip(q, k, v, w)
        ]

        q, k, v = map(
            lambda t: t.view(batch * self.groups, self.num_heads, -1,
                             dim // self.num_heads // self.groups).transpose(1, 2),
            [q, k, v]
        )
        w = w.view(batch * self.groups, self.num_heads, -
                   1, dim // self.num_heads // self.groups)

        attn_output = (q @ k.transpose(-2, -1)) * \
            (dim // self.num_heads // self.groups) ** -0.5
        attn_output = attn_output.softmax(dim=-1)
        attn_output = (attn_output @ v).transpose(1,
                                                  2).reshape(batch, seq_len, dim)
        return self.out(attn_output * w.reshape(batch, seq_len, dim))


class SlidingWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.head_dim ** -0.5

        # Pad to multiple of window size
        padding = (self.window_size - N % self.window_size) % self.window_size
        q = F.pad(q, (0, 0, 0, padding))
        k = F.pad(k, (0, 0, 0, padding))
        v = F.pad(v, (0, 0, 0, padding))

        # Reshape to sliding windows
        q = q.reshape(B * self.num_heads, self.window_size, -1)
        k = k.reshape(B * self.num_heads, self.window_size, -1)
        v = v.reshape(B * self.num_heads, self.window_size, -1)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attn @ v

        attn = attn.reshape(B, self.num_heads, N + padding, self.head_dim)
        attn = attn[:, :, :N, :].permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(attn)


class TinyMoE(nn.Module):
    def __init__(self, dim, num_experts, num_active_experts, expert_dim, dropout=0.0, expert_capacity_scale=1.0, aux_loss_weight=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.expert_dim = expert_dim
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(dim, num_experts)
        self.expert_capacity_scale = expert_capacity_scale
        self.scaled_expert_dim = int(expert_dim * self.expert_capacity_scale)
        self.experts = nn.ModuleList(
            [nn.Linear(dim, self.scaled_expert_dim) for _ in range(num_active_experts)])
        self.fc = nn.Linear(self.scaled_expert_dim, dim)

        # Auxiliary loss
        self.aux_loss_weight = aux_loss_weight
        self.expert_diversity_loss = nn.MSELoss()

    def forward(self, x):
        b, n, d = x.shape

        # Compute attention scores
        scores = self.gate(x).view(b, n, self.num_experts)
        scores = F.softmax(scores, dim=-1)

        # Apply dropout to the attention scores
        scores = self.dropout(scores)

        # Compute the weighted sum of expert outputs
        expert_outputs = torch.stack(
            [exp(x.view(b * n, d)) for exp in self.experts], dim=1)
        expert_outputs = expert_outputs.view(
            b, n, self.num_active_experts, self.scaled_expert_dim)
        weighted_outputs = (
            expert_outputs * scores[:, :, :self.num_active_experts].unsqueeze(-1)).sum(dim=2)

        # Apply the final linear layer
        output = self.fc(weighted_outputs)

        # Auxiliary loss: Expert diversity
        # (b, num_active_experts, scaled_expert_dim)
        expert_activations = expert_outputs.mean(dim=1)
        expert_diversity_loss = self.expert_diversity_loss(expert_activations.transpose(
            0, 1), torch.zeros_like(expert_activations.transpose(0, 1)))

        return output, expert_diversity_loss * self.aux_loss_weight

    def set_expert_capacity(self, expert_capacity_scale):
        self.expert_capacity_scale = expert_capacity_scale
        self.scaled_expert_dim = int(
            self.expert_dim * self.expert_capacity_scale)
        self.experts = nn.ModuleList([nn.Linear(
            self.dim, self.scaled_expert_dim) for _ in range(self.num_active_experts)])
        self.fc = nn.Linear(self.scaled_expert_dim, self.dim)


class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)

        # stay in here because this is a core component
        self.tmix = RWKV_TimeMix_x051a(config, layer_id)

        # Add GroupedQAttention instance
        self.grouped_attn = GroupedQAttention(config.n_embd, config.n_head)

        # stay in here because this is a core component
        self.cmix = RWKV_ChannelMix_x051a(config, layer_id)

        self.sliding_attn = SlidingWindowAttention(
            config.n_embd, window_size=256, num_heads=config.n_head)

        self.moe = TinyMoE(config.dim, config.num_experts, config.num_active_experts,
                           config.expert_dim, config.dropout, expert_capacity_scale=1.2, aux_loss_weight=0.01)

    def forward(self, x):
        x = x + self.tmix(self.ln_1(x))
        x = x + self.cmix(self.ln_2(x))
        x = x + self.sliding_attn(x)  # Apply sliding window attention
        x = x + self.grouped_attn(self.tmix(x))  # Apply GroupedQAttention
        # x = x + self.moe(x)  # Apply TinyMoE
        moe_output, aux_loss = self.moe(x)
        x = x + moe_output
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('tmix.output.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)

        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=None):

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
