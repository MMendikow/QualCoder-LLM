#!/usr/bin/env python3
"""
Usage:
  python run.py configs/config.yaml

This calls the same code path as:
  python -m src.main configs/config.yaml
"""
from src.main import main

if __name__ == "__main__":
    main()
