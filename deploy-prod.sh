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
