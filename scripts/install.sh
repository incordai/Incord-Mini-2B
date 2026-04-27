#!/usr/bin/env bash
# Idempotent install script for QORA API server on Ubuntu 24.04+.
# Run as root. Assumes target/release/qora-server has been built.

set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "must run as root" >&2; exit 1; }

echo "==> Creating qora user and directories..."
id -u qora >/dev/null 2>&1 || \
    useradd --system --no-create-home --shell /usr/sbin/nologin qora
mkdir -p /etc/qora /var/lib/qora /var/log/qora
chown -R qora:qora /var/lib/qora /var/log/qora
chmod 750 /etc/qora /var/lib/qora /var/log/qora

echo "==> Installing binary..."
install -o root -g root -m 0755 target/release/qora-server /usr/local/bin/qora-server

echo "==> Installing config (preserves existing if present)..."
if [ ! -f /etc/qora/qora-server.toml ]; then
    install -o root -g qora -m 0640 \
        config/qora-server.toml.example /etc/qora/qora-server.toml
    echo "    Wrote /etc/qora/qora-server.toml — review and edit before starting."
else
    echo "    /etc/qora/qora-server.toml already exists — preserved."
fi

echo "==> Creating env file template (preserves existing)..."
if [ ! -f /etc/qora/qora.env ]; then
    cat > /etc/qora/qora.env <<'EOF'
# Required: secret used as HMAC key for API key index. Generate via:
#   QORA_SERVER_SECRET=$(head -c 32 /dev/urandom | xxd -p -c 64)
QORA_SERVER_SECRET=CHANGE_ME

# Required: admin token used to create/revoke API keys. Generate via:
#   QORA_ADMIN_TOKEN=$(head -c 32 /dev/urandom | xxd -p -c 64)
QORA_ADMIN_TOKEN=CHANGE_ME

# Optional: rust log level
RUST_LOG=qor4b=info,tower_http=info
EOF
    chown root:qora /etc/qora/qora.env
    chmod 0640 /etc/qora/qora.env
    echo "    Wrote /etc/qora/qora.env — set QORA_SERVER_SECRET and QORA_ADMIN_TOKEN before starting."
else
    echo "    /etc/qora/qora.env already exists — preserved."
fi

echo "==> Installing systemd unit..."
install -o root -g root -m 0644 config/qora-server.service /etc/systemd/system/qora-server.service
systemctl daemon-reload

echo
echo "Installation complete."
echo
echo "Next steps:"
echo "  1. Edit /etc/qora/qora-server.toml to set model paths, slot count, etc."
echo "  2. Edit /etc/qora/qora.env to set QORA_SERVER_SECRET and QORA_ADMIN_TOKEN."
echo "  3. Place model.qor4b and tokenizer.json under /var/lib/qora/ (or wherever the config points)."
echo "  4. Start the service:"
echo "       systemctl enable --now qora-server"
echo "       journalctl -u qora-server -f"
