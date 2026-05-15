# Resultados Experimentais — Reconhecimento de LIBRAS

Este documento descreve todos os experimentos realizados, decisões de design e resultados obtidos no desenvolvimento do sistema de reconhecimento de LIBRAS.

---

## 1. Dataset

### 1.1 Composição

O dataset final possui **55 classes** e combina duas origens:

| Origem | Qtd | Exemplos |
|--------|-----|---------|
| MINDS (acadêmico) | 20 | Acontecer, Aluno, Amarelo, América, Aproveitar, Bala, Banco, Banheiro, Barulho, Cinco, Conhecer, Espelho, Esquina, Filho, Maçã, Medo, Ruim, Sapo, Vacina, Vontade |
| Adicionados (coletados) | 35 | Letras a–z (menos algumas), ajuda, eu, não, por favor, qual, quer, sim, tudo bem, você |

Detalhes completos por sinal (nº de mãos, natureza estática/dinâmica):

| Sinal | Origem | Mãos | Natureza |
|-------|--------|------|---------|
| Acontecer | MINDS | Duas | Dinâmico |
| Aluno | MINDS | Uma | Estático |
| Amarelo | MINDS | Uma | Dinâmico |
| América | MINDS | Duas | Estático |
| Aproveitar | MINDS | Uma | Dinâmico |
| Bala | MINDS | Uma | Dinâmico |
| Banco | MINDS | Uma | Dinâmico |
| Banheiro | MINDS | Duas | Dinâmico |
| Barulho | MINDS | Uma | Dinâmico |
| Cinco | MINDS | Uma | Estático |
| Conhecer | MINDS | Uma | Dinâmico |
| Espelho | MINDS | Uma | Dinâmico |
| Esquina | MINDS | Duas | Dinâmico |
| Filho | MINDS | Uma | Dinâmico |
| Maçã | MINDS | Uma | Dinâmico |
| Medo | MINDS | Uma | Dinâmico |
| Ruim | MINDS | Uma | Dinâmico |
| Sapo | MINDS | Duas | Dinâmico |
| Vacina | MINDS | Uma | Dinâmico |
| Vontade | MINDS | Uma | Dinâmico |
| a | Adicionado | Uma | Estático |
| ajuda | Adicionado | Duas | Dinâmico |
| b | Adicionado | Uma | Estático |
| c | Adicionado | Uma | Estático |
| d | Adicionado | Uma | Estático |
| e | Adicionado | Uma | Estático |
| eu | Adicionado | Uma | Estático |
| f | Adicionado | Uma | Estático |
| g | Adicionado | Uma | Estático |
| h | Adicionado | Uma | Estático |
| i | Adicionado | Uma | Estático |
| j | Adicionado | Uma | Dinâmico |
| k | Adicionado | Uma | Estático |
| l | Adicionado | Uma | Estático |
| m | Adicionado | Uma | Estático |
| n | Adicionado | Uma | Estático |
| não | Adicionado | Uma | Dinâmico |
| o | Adicionado | Uma | Estático |
| p | Adicionado | Uma | Estático |
| por favor | Adicionado | Duas | Dinâmico |
| q | Adicionado | Uma | Estático |
| qual | Adicionado | Uma | Dinâmico |
| quer | Adicionado | Uma | Dinâmico |
| r | Adicionado | Uma | Estático |
| s | Adicionado | Uma | Estático |
| sim | Adicionado | Uma | Dinâmico |
| t | Adicionado | Uma | Estático |
| tudo bem | Adicionado | Duas | Dinâmico |
| u | Adicionado | Uma | Estático |
| v | Adicionado | Uma | Estático |
| você | Adicionado | Uma | Dinâmico |
| w | Adicionado | Uma | Dinâmico |
| x | Adicionado | Uma | Dinâmico |
| y | Adicionado | Uma | Dinâmico |
| z | Adicionado | Uma | Dinâmico |

### 1.2 Features

Cada sequência é representada como uma matriz **(T × 126)**:
- T = número de frames (variável; padronizado para T_fixo durante treino)
- 126 features = 2 mãos × 21 landmarks × 3 coordenadas (x, y, z)
- Normalização **wrist-centered**: subtrai o landmark 0 (pulso) de cada mão → invariância à translação

### 1.3 Split

- 80% treino / 20% teste
- Split por **grupo** (sessão/gravador) para evitar vazamento de dados entre treino e teste
- Seed 42 (determinístico e reproduzível)

---

## 2. Modelos e Arquiteturas

### 2.1 LSTM (baseline)

```
LSTM [128] → LSTM [64] → Dense [128] → Dropout(0.4) → Softmax
```

### 2.2 BiLSTM

```
BiLSTM [128] → BiLSTM [64] → Dense [128] → Dropout(0.4) → Softmax
```

### 2.3 BiLSTM+Atenção (padrão)

```
BiLSTM [160] → BiLSTM [128] + LayerNorm
MultiHeadAttention (4 cabeças, key_dim=32) + conexão residual
GlobalAveragePooling1D
Dense [192] → Dense [128] → Dropout(0.4) → Softmax
```

### 2.4 BiLSTM+Atenção (melhor — `bilstm_attn_best`)

Configuração derivada dos estudos de ablação (seção 4):

```
Entrada: (T=32, F=126) — wrist-centered

BiLSTM [256] + LayerNorm
BiLSTM [192] + LayerNorm
MultiHeadAttention (8 cabeças, key_dim=64) — sem conexão residual
GlobalMaxPooling1D
Dense [64] → Dropout(0.0) → Softmax(55)
```

Parâmetros de treino: label_smooth=0.10, sem class weights, T=32.

---

## 3. Desempenho dos modelos principais

Avaliação no conjunto de teste (80/20 split, 363 amostras de teste, 55 classes):

| Modelo | Acurácia | Precisão (macro) | Recall (macro) | F1 (macro) | Kappa | Épocas |
|--------|----------|-----------------|----------------|------------|-------|--------|
| LSTM | 93,94% | 94,72% | 94,12% | 94,11% | 0,938 | 241 |
| BiLSTM+Atenção (padrão) | ~95,6% | — | — | — | — | — |
| **BiLSTM+Atenção (melhor)** | **97,25%** | **97,72%** | **97,22%** | **97,26%** | **0,972** | 97 |

O modelo melhor convergiu em apenas **97 épocas** (contra 241 do LSTM) graças ao EarlyStopping (patience=15) — a arquitetura mais rica aprende mais rápido e com maior precisão.

Métricas adicionais do bilstm_attn_best:
- Top-3 accuracy: 98,90%
- MCC: 0,972
- ECE (calibração): 0,106

Figuras: `results_paper/fig_curvas_treino.png`, `results_paper/fig1_metricas.png`

---

## 4. Estudos de Ablação

22 suítes independentes variando um hiperparâmetro por vez, com baseline `bilstm_attn`. Todas as avaliações usam bootstrap com 1000 reamostras. Resultados em `results_ablation/`.

### 4.1 Arquitetura

| Suíte | Variação | Melhor valor | Δ Acurácia vs médio |
|-------|----------|-------------|---------------------|
| `arch` | LSTM vs BiLSTM vs BiLSTM+Attn | BiLSTM+Attn | +0.0340 vs LSTM |
| `depth` | 1, 2 ou 3 camadas BiLSTM | 2 camadas | — |
| `units` | small/medium/large | large [256,192] | +0.0083 |
| `attn_heads` | 1, 2, 4, 8 cabeças | 8 cabeças | +0.0081 vs 4 |
| `attn_key_dim` | 16, 32, 64 | 64 | +0.0054 vs 32 |
| `dense_head` | small [64] / medium [192,128] / large [256,128] | small [64] | +0.0081 |
| `pooling` | avg / max / last | max | marginal |
| `layer_norm` | com / sem LayerNorm | com LayerNorm | +0.0082 |
| `residual` | com / sem conexão residual | sem residual | +0.0081 |

### 4.2 Regularização

| Suíte | Variação | Melhor valor | Δ Acurácia |
|-------|----------|-------------|------------|
| `dropout` | 0.0, 0.2, 0.4, 0.6 | **0.0** | **+0.0220** (maior ganho!) |
| `rec_dropout` | 0.00, 0.10, 0.15, 0.30 | 0.15 | — |
| `label_smooth` | 0.00, 0.05, 0.10, 0.20 | 0.10 | +0.0055 |
| `class_weight` | com / sem pesos de classe | sem | marginal |

### 4.3 Treinamento e dados

| Suíte | Variação | Melhor valor | Δ Acurácia |
|-------|----------|-------------|------------|
| `lr_sched` | ReduceLROnPlateau vs CosineDecay | ReduceLROnPlateau | — |
| `optimizer` | AdamW / Adam / SGD | AdamW | — |
| `aug_ind` | augmentações individuais vs todas | todas | +0.0109 |
| `seq_ext` | T = 8, 12, 16, 24, 32 | **T=32** | +0.0081 |
| `data_frac` | 25%, 50%, 75%, 100% dos dados | 100% | curva crescente |

### 4.4 Análise de sinais confundíveis

Suíte `confusable` — análise das confusões mais frequentes no conjunto de teste:

- **i → j e j → i**: par mais confundido. "i" é estático (configuração manual sem movimento); "j" usa a mesma configuração inicial mas adiciona um movimento curvo. O modelo `bilstm_attn_combined` confundiu j→i com confiança de 98,41%, indicando que o movimento dinâmico de J é difícil de distinguir no início da sequência.
- Letras com configuração manual similar (h/u, o/c) também aparecem entre as confusões.

Figura detalhada: `results_paper/fig5_ij_analise.png`

---

## 5. Estudo de Generalização MINDS

Avalia como os modelos transferem entre o dataset customizado e o MINDS. Resultados em `results_MINDS/`.

### 5.1 Os três modelos

| Modelo | Treinado em | Classes |
|--------|-------------|---------|
| `bilstm_attn` | dataset/ | 55 (custom) |
| `bilstm_attn_minds` | libras_data_minds/ | 20 (MINDS) |
| `bilstm_attn_combined` | dataset/ + libras_data_minds/ | 55 (ambos) |

### 5.2 Seis cenários de avaliação

| ID | Modelo | Avaliado em | N | Classes | Acc | F1 | Prec | Rec |
|----|--------|-------------|---|---------|-----|-----|------|-----|
| A | bilstm_attn | dataset/ | 1808 | 55 | 98,56% | 0,9861 | 0,9872 | 0,9859 |
| B | bilstm_attn | libras_data_minds/ | 800 | 20 | 43,87% | 0,3338 | 0,2928 | 0,4388 |
| C | bilstm_attn_minds | libras_data_minds/ | 800 | 20 | **97,75%** | **0,9774** | 0,9783 | 0,9775 |
| D | bilstm_attn_minds | dataset/ (20 cls) | 690 | 20 | 73,19% | 0,6392 | 0,7143 | 0,6946 |
| E | bilstm_attn_combined | dataset/ | 1808 | 55 | **98,67%** | **0,9864** | 0,9866 | 0,9868 |
| F | bilstm_attn_combined | libras_data_minds/ | 800 | 20 | 95,00% | 0,9499 | 0,9525 | 0,9500 |

### 5.3 Análise dos cenários

**Cenário B (43,87%)** — Zero-shot: o modelo treinado somente no custom falha completamente no MINDS. Os domínios têm distribuições diferentes (iluminação, sinalante, enquadramento).

**Cenário C vs F** — MINDS-only (97,75%) supera Combined (95,00%) no próprio MINDS. O modelo combinado perde ~2,8 pp porque divide capacidade entre 55 classes. Porém, o combined ainda cobre as 55 classes com 98,67% no custom (cenário E), enquanto o MINDS-only cobre apenas 20.

**Cenário E ≥ A** — Adicionar dados MINDS não prejudica o custom: 98,67% > 98,56%.

**Conclusão:** Para uso em produção cobrindo todos os 55 sinais, `bilstm_attn_combined` é o melhor modelo. Se o uso for restrito aos 20 sinais MINDS, `bilstm_attn_minds` tem vantagem.

Figuras: `results_paper/fig_minds_comparacao.png`, `results_MINDS/minds_f1_per_class.png`

---

## 6. Eficiência Computacional

Medido em CPU pura (TF 2.16.1, sem GPU), batch=1, 200 inferências com 20 warm-up.  
GFLOPs calculados analiticamente por camada (multiply-add = 2 FLOPs).

| Modelo | Parâmetros | FP32 | GFLOPs/seq | Lat. média | Lat. p95 | Pipeline web |
|--------|-----------|------|-----------|-----------|---------|-------------|
| `lstm` | 195.383 | 0,76 MB | 0,0058 | 64,26 ms | 73,03 ms | 62,90 ms |
| `bilstm_attn` | 1.041.655 | 3,97 MB | 0,0309 | 304,32 ms | 320,85 ms | 317,97 ms |
| `bilstm_attn_best` | 2.686.391 | 10,25 MB | 0,1719 | 548,46 ms | 570,05 ms | 547,74 ms |
| `bilstm_attn_minds` | 1.037.140 | 3,96 MB | 0,0309 | 320,14 ms | 328,60 ms | 324,19 ms |
| `bilstm_attn_combined` | 1.041.655 | 3,97 MB | 0,0309 | 321,72 ms | 337,18 ms | 321,35 ms |

**Legenda:** Lat. = latência da chamada ao modelo (batch=1, CPU) · Pipeline web = feature transform + z-score + modelo + EMA (simula `Predictor.step()`)

### Detalhamento do pipeline web — BiLSTM+Atenção combinado

| Etapa | Média | p95 |
|-------|-------|-----|
| Feature transform (wrist-centered) | 0,038 ms | 0,052 ms |
| Normalização z-score | 0,040 ms | 0,055 ms |
| **Inferência do modelo** | **321,35 ms** | **330,71 ms** |
| EMA update | 0,023 ms | 0,032 ms |
| **Total** | **321,72 ms** | **337,18 ms** |

O modelo domina >99,9% do tempo de pipeline; o overhead de pré-processamento e pós-processamento é desprezível (<0,15 ms).

### Observações

**LSTM vs BiLSTM+Attn (padrão):** LSTM tem 5,3× menos parâmetros (195 K vs 1,04 M) e é ~4,7× mais rápido (64 ms vs 304 ms), com acurácia 3,3 pp inferior (93,94% vs 97,25%).

**bilstm_attn_best vs bilstm_attn:** O modelo otimizado tem 2,6× mais parâmetros e usa T=32 (vs T=16), resultando em ~29,6× mais GFLOPs (0,1719 vs 0,0058) e ~1,8× mais latência (548 ms vs 304 ms), com ganho de 1,7 pp de acurácia.

**Impacto do `INFER_EVERY_N=2`:** A aplicação web executa inferência a cada 2 frames — ao custo de uma latência percebida de ~640 ms no bilstm_attn_combined (2 frames × 30 fps ≈ 66 ms de espera + 322 ms de inferência), o modelo responde dentro de ~1 ciclo de câmera mobile (~33 ms por frame).

**Tradeoff para produção:** `bilstm_attn_combined` oferece o melhor equilíbrio — mesma arquitetura e latência do `bilstm_attn` (304–322 ms), mas com +2,7 pp de acurácia e cobertura de 55 classes incluindo os 20 sinais MINDS.

Arquivo completo: `results_paper/efficiency_results.json` · `results_paper/efficiency_results.csv`

---

## 7. Figuras para publicação

Todas em `results_paper/` (PNG 300 DPI + PDF vetorial):

| Arquivo | Descrição |
|---------|-----------|
| `fig1_metricas.png` | Acc/Prec/Rec/F1 com eixo cortado — LSTM vs Combined |
| `fig2_confiancas.png` | Histograma de confiança softmax: acertos vs erros |
| `fig3_pr_curves.png` | Curvas Precisão-Revocação macro (com mAP) |
| `fig4_f1_categorias.png` | F1 por categoria (1/2 mãos × estático/dinâmico) |
| `fig5_ij_analise.png` | Análise do par confundível I (estático) vs J (dinâmico) |
| `fig_curvas_treino.png` | Evolução de Loss e Acurácia por época — LSTM vs bilstm_attn_best |
| `fig_minds_comparacao.png` | MINDS-only vs Combined nas 20 classes MINDS |
| `tabela_resultados.tex` | Tabela LaTeX pronta para incluir no artigo |

---

## 8. Reprodutibilidade

Todos os experimentos usam `SEED=42` (numpy, tensorflow, random). O split treino/teste é determinístico com `USE_GROUP_SPLIT=True`.

Para reproduzir o modelo principal:
```bash
py -3.11 train.py --model bilstm_attn_best --label_smooth 0.10 --no_weights --T 32
```

Para reproduzir o estudo MINDS completo:
```bash
py -3.11 minds_study.py --epochs 80
```

Para reproduzir toda a ablação (~6h):
```bash
py -3.11 ablation.py --suite all
```
