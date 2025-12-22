#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Activate venv
venv_path = os.path.join(project_dir, "venv", "bin", "activate_this.py")
if os.path.exists(venv_path):
    exec(open(venv_path).read(), {"__file__": venv_path})

# Load Flask application
from app import app as application
