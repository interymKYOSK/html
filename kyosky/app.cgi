#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CGI wrapper for Flask app on Uberspace
"""
import sys
import os

# Add the application directory to sys.path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

from wsgiref.handlers import CGIHandler
from app import app

if __name__ == "__main__":
    CGIHandler().run(app)
