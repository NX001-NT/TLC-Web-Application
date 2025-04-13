#!/bin/bash

# Step 1: Install Python
echo "Installing Python..."
sudo apt-get update && sudo apt-get install -y python3 python3-pip

# Step 2: Install Python packages (adjust the path if needed)
echo "Installing Python dependencies..."
pip3 install -r TLC-Final-Python/requirements.txt

# Step 3: Install Node.js packages
echo "Installing Node dependencies..."
npm install

# Step 4: Start your Node.js server
echo "Starting Node server..."
node index.js
