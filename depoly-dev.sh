#!/bin/bash

echo "🚀 Deploying yolo-dev.service..."

# Clean local Python cache
rm -rf __pycache__/

# Copy dev service file
sudo cp yolo-dev.service /etc/systemd/system/

# Restart dev service
sudo systemctl daemon-reload
sudo systemctl restart yolo-dev.service
sudo systemctl enable yolo-dev.service

# Check status
echo "📊 Service status:"
sudo systemctl status yolo-dev.service --no-pager
