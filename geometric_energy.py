#!/usr/bin/env python3
"""
GEOMETRIC vs VANILLA ENERGY BENCHMARK
© 2026 Eric Waller, Patent Pending

Measures Tok/J (tokens per joule) - the key metric for energy efficiency.

Results on H100:
  GEOMETRIC: 523W | 15.14ms | 2,070 Tok/J
  VANILLA  : 584W | 17.93ms | 1,564 Tok/J
  Efficiency: +32.3% | Power: -61W | Latency: -15.5%
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import subprocess

def get_power():
    r = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                       capture_output=True, text=True)
    return float(r.stdout.strip().split('\n')[0])

D_MODEL, D_MLP, N_LAYERS = 768, 3072, 12

class GeometricMLP(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.W_fc_active = nn.Linear(d_model, d_model, bias=True)
        self.W_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, payload=None):
        h_active = F.gelu(self.W_fc_active(x))
        out = self.W_proj(h_active) + x
        return out, payload

class VanillaMLP(nn.Module):
    def __init__(self, d_model=D_MODEL, d_mlp=D_MLP):
        super().__init__()
        self.fc = nn.Linear(d_model, d_mlp)
        self.proj = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x))) + x

class GeometricBlock(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, 12, batch_first=True)
        self.mlp = GeometricMLP(d_model)

    def forward(self, x, payload=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        out, payload = self.mlp(self.ln2(x), payload)
        return out, payload

class VanillaBlock(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, 12, batch_first=True)
        self.mlp = VanillaMLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        return x + self.mlp(self.ln2(x))

class GeometricTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS):
        super().__init__()
        self.blocks = nn.ModuleList([GeometricBlock(d_model) for _ in range(n_layers)])

    def forward(self, x, payload=None):
        for block in self.blocks:
            x, payload = block(x, payload)
        return x, payload

class VanillaTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS):
        super().__init__()
        self.blocks = nn.ModuleList([VanillaBlock(d_model) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def bench(name, model, x, iters=100, use_payload=False):
    torch.cuda.synchronize()
    p0, t0 = get_power(), time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            if use_payload:
                _ = model(x, None)
            else:
                _ = model(x)
    torch.cuda.synchronize()
    p1, t1 = get_power(), time.perf_counter()
    dt = t1 - t0
    pwr = (p0 + p1) / 2
    tok = x.shape[0] * x.shape[1] * iters
    tpj = tok / (pwr * dt)
    ms = (dt / iters) * 1000
    print(f"{name}: {pwr:.0f}W | {ms:.2f}ms | {tok/dt:,.0f} tok/s | {tpj:,.0f} Tok/J")
    return tpj, pwr, ms

if __name__ == "__main__":
    print("=" * 60)
    print("GEOMETRIC vs VANILLA - ENERGY TEST")
    print("=" * 60)
    geo = GeometricTransformer().cuda().half().eval()
    van = VanillaTransformer().cuda().half().eval()
    print(f"Geometric: {sum(p.numel() for p in geo.parameters())/1e6:.1f}M params")
    print(f"Vanilla:   {sum(p.numel() for p in van.parameters())/1e6:.1f}M params")
    x = torch.randn(32, 512, D_MODEL, device='cuda', dtype=torch.half)
    with torch.inference_mode():
        for _ in range(20):
            geo(x, None); van(x)
    torch.cuda.synchronize()
    print()
    g_tpj, g_pwr, g_ms = bench("GEOMETRIC", geo, x, use_payload=True)
    v_tpj, v_pwr, v_ms = bench("VANILLA  ", van, x)
    print()
    print(f"Efficiency: {((g_tpj/v_tpj)-1)*100:+.1f}% | Power: {g_pwr-v_pwr:+.0f}W | Latency: {((g_ms/v_ms)-1)*100:+.1f}%")
