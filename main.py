#!/usr/bin/env python3
"""
VectorFlow Entry Point
Wrapper script to run the VectorFlow CLI from the project root.
"""

import sys
import os

# Add project root to path to ensure vectorflow package is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectorflow.cli import main

if __name__ == "__main__":
    main()
