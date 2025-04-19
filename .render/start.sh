#!/bin/bash

# Create a Python virtual environment (if needed)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies in the virtual environment
pip install --upgrade pip
pip install -r requirements.txt

# Start your server
npm install
npm install express-fileupload
node index2.js
