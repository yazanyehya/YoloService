#!/bin/bash

echo "🚀 Deploying yolo.service..."

# Clean local Python cache
rm -rf __pycache__/

# Copy service file
sudo cp yolo.service /etc/systemd/system/

# Restart service
sudo systemctl daemon-reload
sudo systemctl restart yolo.service
sudo systemctl enable yolo.service

# Check status
echo "📊 Service status:"
sudo systemctl status yolo.service --no-pager
