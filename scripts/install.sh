#!/usr/bin/env bash
# Idempotent install script for Incord API server on Ubuntu 24.04+.
# Run as root. Assumes target/release/incord-server has been built.

set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "must run as root" >&2; exit 1; }

echo "==> Creating incord user and directories..."
id -u incord >/dev/null 2>&1 || \
    useradd --system --no-create-home --shell /usr/sbin/nologin incord
mkdir -p /etc/incord /var/lib/incord /var/log/incord
chown -R incord:incord /var/lib/incord /var/log/incord
chmod 750 /etc/incord /var/lib/incord /var/log/incord

echo "==> Installing binary..."
install -o root -g root -m 0755 target/release/incord-server /usr/local/bin/incord-server

echo "==> Installing config (preserves existing if present)..."
if [ ! -f /etc/incord/incord-server.toml ]; then
    install -o root -g incord -m 0640 \
        config/incord-server.toml.example /etc/incord/incord-server.toml
    echo "    Wrote /etc/incord/incord-server.toml — review and edit before starting."
else
    echo "    /etc/incord/incord-server.toml already exists — preserved."
fi

echo "==> Creating env file template (preserves existing)..."
if [ ! -f /etc/incord/incord.env ]; then
    cat > /etc/incord/incord.env <<'EOF'
# Required: secret used as HMAC key for API key index. Generate via:
#   INCORD_SERVER_SECRET=$(head -c 32 /dev/urandom | xxd -p -c 64)
INCORD_SERVER_SECRET=CHANGE_ME

# Required: admin token used to create/revoke API keys. Generate via:
#   INCORD_ADMIN_TOKEN=$(head -c 32 /dev/urandom | xxd -p -c 64)
INCORD_ADMIN_TOKEN=CHANGE_ME

# Optional: rust log level
RUST_LOG=incord=info,tower_http=info
EOF
    chown root:incord /etc/incord/incord.env
    chmod 0640 /etc/incord/incord.env
    echo "    Wrote /etc/incord/incord.env — set INCORD_SERVER_SECRET and INCORD_ADMIN_TOKEN before starting."
else
    echo "    /etc/incord/incord.env already exists — preserved."
fi

echo "==> Installing systemd unit..."
install -o root -g root -m 0644 config/incord-server.service /etc/systemd/system/incord-server.service
systemctl daemon-reload

echo
echo "Installation complete."
echo
echo "Next steps:"
echo "  1. Edit /etc/incord/incord-server.toml to set model paths, slot count, etc."
echo "  2. Edit /etc/incord/incord.env to set INCORD_SERVER_SECRET and INCORD_ADMIN_TOKEN."
echo "  3. Place model.incord and tokenizer.json under /var/lib/incord/ (or wherever the config points)."
echo "  4. Start the service:"
echo "       systemctl enable --now incord-server"
echo "       journalctl -u incord-server -f"
