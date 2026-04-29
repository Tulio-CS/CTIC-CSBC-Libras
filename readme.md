# LSB Object Detection — LIBRAS Recognition

Reconhecimento de gestos LIBRAS em tempo real com MediaPipe + BiLSTM + Attention.

---

## Estrutura do projeto

```
LSB_Object_Detection/
├── config.py          # Hiperparâmetros e caminhos centralizados
├── data_utils.py      # Carregamento, normalização, augmentação, métricas
├── models.py          # Arquiteturas (LSTM, BiLSTM, BiLSTM+Attention)
├── train.py           # Treinamento unificado
├── evaluate.py        # Avaliação completa + gráficos
├── collect_data.py    # Coleta de sequências via webcam
├── infer_live.py      # Inferência ao vivo via webcam (desktop)
├── benchmark.py       # Mede latência por etapa
├── process_minds.py   # Converte dataset MINDS para formato .npy
├── models/            # Modelos treinados
│   └── bilstm_attn/
│       ├── model.keras
│       ├── norm_stats.json
│       └── actions.npy
└── web/               # Servidor web para inferência no celular
    ├── app.py
    ├── predictor.py
    ├── requirements.txt
    ├── nginx.conf
    ├── deploy.sh
    └── frontend/
        └── index.html
```

---

## Fluxo de trabalho

### 1. Coletar dados
```bash
py -3.11 collect_data.py --sign Ola --sequences 30
py -3.11 collect_data.py --sign Obrigado --sequences 30
```
Teclas: `SPACE` = iniciar sequência · `R` = regravar · `Q` = sair

### 2. Treinar modelo
```bash
py -3.11 train.py --model bilstm_attn --data libras_data --epochs 80
```
O modelo é salvo em `models/bilstm_attn/`.

### 3. Testar ao vivo (desktop)
```bash
py -3.11 infer_live.py --model bilstm_attn
```

### 4. Avaliar métricas
```bash
py -3.11 evaluate.py --model bilstm_attn
```

---

## Deploy na VPS (servidor web para celular)

### Pré-requisitos na VPS
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv nginx
```

### Enviar projeto para a VPS
```bash
# Na sua máquina local — envie apenas o necessário
rsync -av --exclude='old/' --exclude='*.pyc' --exclude='__pycache__' \
  --exclude='libras_data/' --exclude='libras_data_minds/' \
  ./ user@IP_DA_VPS:~/LSB_Object_Detection/
```
> Certifique-se de incluir a pasta `models/bilstm_attn/` com os 3 arquivos.

### Executar o deploy
```bash
# Na VPS
cd ~/LSB_Object_Detection/web
chmod +x deploy.sh
./deploy.sh
```

O script automaticamente:
1. Cria um virtualenv Python 3.11 em `web/.venv/`
2. Instala as dependências (`tensorflow-cpu`, `fastapi`, etc.)
3. Cria um serviço **systemd** (`libras.service`) que inicia o uvicorn automaticamente
4. Configura o **nginx** como reverse proxy (porta 80)

### Ativar HTTPS (obrigatório para câmera no celular)
```bash
./deploy.sh --ssl seu-dominio.com
```
Isso instala o certbot e obtém um certificado Let's Encrypt, ativando HTTPS.
Sem HTTPS, navegadores mobile bloqueiam acesso à câmera.

### Comandos úteis na VPS
```bash
# Status do serviço
sudo systemctl status libras

# Ver logs em tempo real
sudo journalctl -u libras -f

# Reiniciar após update do modelo
./deploy.sh --update

# Verificar se a API está respondendo
curl http://localhost:8000/health
```

### Acesso
Abra no celular: `https://seu-dominio.com`

O site detecta gestos LIBRAS em tempo real usando a câmera do celular. Não requer instalação de app.

---

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `LIBRAS_MODEL` | `bilstm_attn` | Nome do modelo em `models/` |

```bash
# Para usar outro modelo no servidor
LIBRAS_MODEL=bilstm uvicorn web.app:app --host 0.0.0.0 --port 8000
```

---

## Protocolo WebSocket

O frontend envia por frame:
```json
{ "landmarks": [126 floats], "reset": false }
```

O servidor responde por frame:
```json
{
  "pred": "Ola",
  "conf": 0.95,
  "top3": { "Ola": 0.95, "Nao": 0.03, "Sim": 0.02 },
  "buffer_fill": 15,
  "buffer_need": 15
}
```

A predição só aparece quando `buffer_fill == buffer_need` (janela cheia).
