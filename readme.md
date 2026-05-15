# Reconhecimento de LIBRAS — BiLSTM + Atenção

Reconhecimento de gestos da Língua Brasileira de Sinais (LIBRAS) em tempo real utilizando landmarks de mãos (MediaPipe) e redes BiLSTM com mecanismo de atenção.

---

## Modelos disponíveis

| Nome | Dados de treino | Classes | Acurácia |
|------|----------------|---------|----------|
| `lstm` | dataset/ | 55 | 93,94% |
| `bilstm_attn` | dataset/ | 55 | ~95,6% |
| `bilstm_attn_best` | dataset/ | 55 | **97,25%** |
| `bilstm_attn_minds` | libras_data_minds/ | 20 | 97,75% (no MINDS) |
| `bilstm_attn_combined` | dataset/ + libras_data_minds/ | 55 | 98,07% |

- **`bilstm_attn_minds`** — treinado **somente no dataset MINDS** (20 sinais de `libras_data_minds/`)
- **`bilstm_attn_combined`** — treinado no dataset custom + MINDS combinados (55 classes, com dados extras para os 20 sinais MINDS)

Para inferência ao vivo use `bilstm_attn_best` ou `bilstm_attn_combined`. Para uso exclusivo com os 20 sinais MINDS use `bilstm_attn_minds`.

---

## Estrutura do projeto

```
LSB_Object_Detection/
├── config.py             # Hiperparâmetros e caminhos centralizados
├── data_utils.py         # Carregamento, normalização, augmentação, métricas
├── models.py             # Arquiteturas (LSTM, BiLSTM, BiLSTM+Atenção)
├── train.py              # Treinamento com todos os hiperparâmetros via CLI
├── evaluate.py           # Avaliação completa + geração de gráficos
├── ablation.py           # Estudos de ablação (22 suítes)
├── minds_study.py        # Estudo de generalização no dataset MINDS
├── paper_figures.py      # Figuras para publicação (artigo)
├── collect_data.py       # Coleta de sequências via webcam
├── infer_live.py         # Inferência ao vivo via webcam
├── process_minds.py      # Converte vídeos MINDS brutos (.mp4) para .npy
├── sign_metadata.json    # Metadados dos sinais (mãos, movimento, origem)
│
├── dataset/              # Dataset principal — Classe/*.npy  (55 classes, coletado manualmente)
├── dataset_minds/        # Dataset MINDS processado — Classe/*.npy  (20 classes acadêmicas)
│
├── models/               # Modelos treinados (cada pasta: model.keras + norm_stats.json + actions.npy)
│   ├── bilstm_attn_best/     ← melhor modelo geral
│   ├── bilstm_attn_combined/ ← cobertura máxima (MINDS + custom)
│   ├── bilstm_attn_minds/    ← somente MINDS (20 classes)
│   ├── bilstm_attn/          ← baseline BiLSTM+Attn
│   └── lstm/                 ← baseline LSTM
│
├── results/              # Avaliações dos modelos principais
├── results_ablation/     # Resultados das 22 suítes de ablação
├── results_MINDS/        # Estudo de generalização MINDS (6 cenários)
└── results_paper/        # Figuras e tabelas para publicação (PNG + PDF + CSV + LaTeX)
```

---

## Fluxo de trabalho

### 1. Coletar novos sinais
```bash
py -3.11 collect_data.py --sign NovaSinal --sequences 30
```
Teclas durante a coleta: `SPACE` = gravar · `R` = regravar · `Q` = sair

Saída: `dataset/NovaSinal/*.npy`

### 2. Treinar modelo

```bash
# Modelo padrão (bilstm_attn)
py -3.11 train.py

# Melhor configuração (recomendado para produção)
py -3.11 train.py --model bilstm_attn_best --label_smooth 0.10 --no_weights --T 32

# LSTM baseline
py -3.11 train.py --model lstm

# Combinado com dataset MINDS
py -3.11 train.py --model bilstm_attn --run_name bilstm_attn_combined \
    --extra_data libras_data_minds
```

### 3. Avaliar modelo
```bash
py -3.11 evaluate.py --model bilstm_attn_best
```
Gera em `results/bilstm_attn_best/`: matriz de confusão, curvas ROC/PR, F1 por classe, calibração, t-SNE, histórico de treinamento.

### 4. Inferência ao vivo (webcam)
```bash
py -3.11 infer_live.py --model bilstm_attn_best
```

### 5. Estudos e figuras
```bash
# Ablação — suíte individual ou todas
py -3.11 ablation.py --suite dropout
py -3.11 ablation.py --suite all        # demora várias horas

# Estudo MINDS (treina bilstm_attn_minds e bilstm_attn_combined se não existirem)
py -3.11 minds_study.py --epochs 80

# Figuras de publicação (PNG + PDF em results_paper/)
py -3.11 paper_figures.py
```

---

## Arquitetura BiLSTM+Atenção — melhor configuração

```
Entrada: (T=32, F=126)
  └─ 126 features = 2 mãos × 21 landmarks × 3 coordenadas (x, y, z)
  └─ normalização wrist-centered (subtrai landmark 0 de cada mão)

2 × BiLSTM [256, 192 unidades] + LayerNormalization
MultiHeadAttention (8 cabeças, key_dim=64) — sem conexão residual
GlobalMaxPooling1D
Dense [64] → Dropout(0.0) → Softmax(55 classes)
```

Augmentações (só no treino): jitter gaussiano (σ=0.01), rotação 2D (±8°),
escala espacial (0.9–1.1×), temporal dropout (p=0.05), time masking (10% dos timesteps).

Otimizador: AdamW (lr=3×10⁻⁴, weight_decay=1×10⁻⁴).
Label smoothing: ε=0.10. Sem class weights.
EarlyStopping: patience=15 em val_loss.

---

## Datasets

Dois diretórios de dados, com papéis distintos:

| Diretório | Classes | Origem | Uso |
|-----------|---------|--------|-----|
| `dataset/` | 55 | Coleta manual (webcam) | Treino/avaliação principal |
| `dataset_minds/` | 20 | [MINDS](https://github.com/...) processado com `process_minds.py` | Estudos de generalização cross-dataset |

Os 20 sinais do MINDS são um subconjunto dos 55 de `dataset/` — permitem comparar modelos treinados em dados acadêmicos vs dados coletados.
Detalhes por sinal (nº de mãos, movimento, origem): [`sign_metadata.json`](sign_metadata.json).

---

## Deploy web (inferência no celular)

```bash
# Enviar para VPS
rsync -av --exclude='old/' --exclude='*.pyc' --exclude='__pycache__' \
  --exclude='newMINDS/' ./ user@IP_DA_VPS:~/LSB_Object_Detection/

# Na VPS — HTTPS obrigatório para câmera mobile
cd ~/LSB_Object_Detection/web
./deploy.sh --ssl seu-dominio.com
```

---

Para resultados completos de todos os experimentos: **[RESULTS.md](RESULTS.md)**
