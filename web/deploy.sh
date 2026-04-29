#!/usr/bin/env bash
# web/deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
# Deploy automatizado do servidor LIBRAS em uma VPS Ubuntu 22.04 / 24.04.
#
# Uso:
#   chmod +x deploy.sh
#   ./deploy.sh                          # primeira vez (instala tudo)
#   ./deploy.sh --update                 # atualiza código e reinicia serviço
#   ./deploy.sh --ssl seu-dominio.com    # ativa HTTPS com Let's Encrypt
#
# Pré-requisitos na VPS:
#   • Ubuntu 22.04 ou 24.04
#   • Python 3.11  (sudo apt install python3.11 python3.11-venv)
#   • nginx        (sudo apt install nginx)
#   • certbot      (opcional, para HTTPS)
#   • O modelo treinado deve estar em:
#       <PROJECT_ROOT>/models/<MODEL_NAME>/model.keras
#       <PROJECT_ROOT>/models/<MODEL_NAME>/norm_stats.json
#       <PROJECT_ROOT>/models/<MODEL_NAME>/actions.npy
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ──────────────── Configurações ──────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # raiz do projeto
WEB_DIR="$PROJECT_ROOT/web"
VENV_DIR="$WEB_DIR/.venv"
SERVICE_NAME="libras"
NGINX_SITE="/etc/nginx/sites-available/$SERVICE_NAME"
MODEL_NAME="${LIBRAS_MODEL:-bilstm_attn}"
PORT=8000
PYTHON="python3.11"

# ──────────────── Cores para output ──────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ──────────────── Argumentos ─────────────────────────────────────────────────
MODE="install"
DOMAIN=""

for arg in "$@"; do
    case "$arg" in
        --update) MODE="update" ;;
        --ssl)    MODE="ssl";   shift; DOMAIN="${1:-}" ;;
    esac
done

# ──────────────── Helpers ─────────────────────────────────────────────────────
need_sudo() {
    if [[ $EUID -ne 0 ]]; then
        warn "Alguns passos precisam de sudo."
    fi
}

check_python() {
    if ! command -v "$PYTHON" &>/dev/null; then
        error "Python 3.11 não encontrado. Instale com: sudo apt install python3.11 python3.11-venv"
    fi
    info "Python: $($PYTHON --version)"
}

check_model() {
    local model_dir="$PROJECT_ROOT/models/$MODEL_NAME"
    for f in model.keras norm_stats.json actions.npy; do
        if [[ ! -f "$model_dir/$f" ]]; then
            error "Arquivo de modelo não encontrado: $model_dir/$f\n       Treine primeiro com: python train.py --model $MODEL_NAME"
        fi
    done
    info "Modelo '$MODEL_NAME' encontrado em $model_dir"
}

create_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        info "Criando ambiente virtual em $VENV_DIR ..."
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet
}

install_deps() {
    info "Instalando dependências Python ..."
    pip install -r "$WEB_DIR/requirements.txt" --quiet
    info "Dependências instaladas."
}

write_systemd_service() {
    info "Criando serviço systemd: $SERVICE_NAME ..."
    sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null <<EOF
[Unit]
Description=LIBRAS Live Inference Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
Environment="LIBRAS_MODEL=$MODEL_NAME"
ExecStart=$VENV_DIR/bin/uvicorn web.app:app --host 127.0.0.1 --port $PORT --workers 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    info "Serviço systemd criado e habilitado."
}

start_or_restart_service() {
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        info "Reiniciando serviço $SERVICE_NAME ..."
        sudo systemctl restart "$SERVICE_NAME"
    else
        info "Iniciando serviço $SERVICE_NAME ..."
        sudo systemctl start "$SERVICE_NAME"
    fi
    sleep 2
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        info "Serviço $SERVICE_NAME está rodando."
    else
        error "Serviço $SERVICE_NAME falhou ao iniciar. Veja: sudo journalctl -u $SERVICE_NAME -n 50"
    fi
}

configure_nginx() {
    if [[ ! -f "$NGINX_SITE" ]]; then
        info "Instalando configuração nginx ..."
        sudo cp "$WEB_DIR/nginx.conf" "$NGINX_SITE"
        sudo ln -sf "$NGINX_SITE" "/etc/nginx/sites-enabled/$SERVICE_NAME"
        # Remove default site se existir
        [[ -f /etc/nginx/sites-enabled/default ]] && sudo rm -f /etc/nginx/sites-enabled/default
    fi
    sudo nginx -t
    sudo systemctl reload nginx
    info "Nginx configurado e recarregado."
}

configure_ssl() {
    [[ -z "$DOMAIN" ]] && error "Informe o domínio: ./deploy.sh --ssl seu-dominio.com"
    if ! command -v certbot &>/dev/null; then
        info "Instalando certbot ..."
        sudo apt-get install -y certbot python3-certbot-nginx
    fi
    info "Obtendo certificado SSL para $DOMAIN ..."
    # Atualiza server_name no nginx antes de rodar o certbot
    sudo sed -i "s/server_name _;/server_name $DOMAIN;/" "$NGINX_SITE"
    sudo nginx -t && sudo systemctl reload nginx
    sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
         --email "admin@$DOMAIN" --redirect
    info "HTTPS ativo para $DOMAIN"
    info ""
    info "IMPORTANTE: câmera no celular só funciona em HTTPS."
    info "Abra:  https://$DOMAIN"
}

show_status() {
    echo ""
    info "══════════════════════════════════════════"
    info "  Deploy concluído!"
    info ""
    info "  Serviço : sudo systemctl status $SERVICE_NAME"
    info "  Logs    : sudo journalctl -u $SERVICE_NAME -f"
    info "  Health  : curl http://127.0.0.1:$PORT/health"
    if [[ -n "$DOMAIN" ]]; then
        info "  URL     : https://$DOMAIN"
    else
        SERVER_IP=$(hostname -I | awk '{print $1}')
        info "  URL     : http://$SERVER_IP"
        warn "Câmera no celular requer HTTPS. Rode: ./deploy.sh --ssl seu-dominio.com"
    fi
    info "══════════════════════════════════════════"
}

# ──────────────── Fluxo principal ────────────────────────────────────────────

case "$MODE" in

    install)
        need_sudo
        check_python
        check_model
        create_venv
        install_deps
        write_systemd_service
        start_or_restart_service
        configure_nginx
        show_status
        ;;

    update)
        info "Modo update: atualizando dependências e reiniciando ..."
        check_model
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        install_deps
        start_or_restart_service
        info "Update concluído."
        ;;

    ssl)
        configure_ssl
        ;;

    *)
        error "Modo desconhecido: $MODE"
        ;;
esac
