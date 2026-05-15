## Eficiência Computacional

Medido em CPU (sem GPU), batch=1, 200 runs com 20 warm-up.

| Modelo | Parâmetros | Tamanho FP32 | GFLOPs/seq | Latência média | Latência p95 | Pipeline web |
|--------|-----------|-------------|-----------|--------------|-------------|-------------|
| `lstm` | 195,383 | 0.8 MB | 0.0058 | 64.26 ms | 73.03 ms | 62.90 ms |
| `bilstm_attn` | 1,041,655 | 4.2 MB | 0.0309 | 304.32 ms | 320.85 ms | 317.97 ms |
| `bilstm_attn_best` | 2,686,391 | 10.7 MB | 0.1719 | 548.46 ms | 570.05 ms | 547.74 ms |
| `bilstm_attn_minds` | 1,037,140 | 4.1 MB | 0.0309 | 320.14 ms | 328.60 ms | 324.19 ms |
| `bilstm_attn_combined` | 1,041,655 | 4.2 MB | 0.0309 | 321.72 ms | 337.18 ms | 321.35 ms |

_Pipeline web = feature transform + normalização + modelo + EMA._
