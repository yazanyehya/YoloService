#!/bin/bash

echo "🚀 Deploying yolo-prod.service..."

# Clean local Python cache
rm -rf __pycache__/

# Copy prod service file
sudo cp yolo-prod.service /etc/systemd/system/

# Restart prod service
sudo systemctl daemon-reload
sudo systemctl restart yolo-prod.service
sudo systemctl enable yolo-prod.service

# Check status
echo "📊 Service status:"
sudo systemctl status yolo-prod.service --no-pager

# -----------------------------
# 📡 OpenTelemetry Collector Setup
# -----------------------------
echo "📡 Setting up OpenTelemetry Collector..."

# Install otelcol if not already installed
if ! command -v otelcol &> /dev/null
then
  echo "📥 Installing OpenTelemetry Collector..."
  wget -q https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.98.0/otelcol_0.98.0_linux_amd64.deb
  sudo dpkg -i otelcol_0.98.0_linux_amd64.deb
  rm otelcol_0.98.0_linux_amd64.deb
else
  echo "✅ otelcol already installed"
fi

# Write otelcol config
sudo tee /etc/otelcol/config.yaml > /dev/null <<EOF
receivers:
  hostmetrics:
    collection_interval: 15s
    scrapers:
      cpu:
      memory:
      disk:
      filesystem:
      load:
      network:
      processes:

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    metrics:
      receivers: [hostmetrics]
      exporters: [prometheus]
EOF

# Restart and enable otelcol
sudo systemctl restart otelcol
sudo systemctl enable otelcol

echo "✅ OpenTelemetry Collector is running and exposing metrics on :8889"
