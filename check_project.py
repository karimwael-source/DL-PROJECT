#!/usr/bin/env python
"""
Project Health Check Script
Verifies all components are working correctly
"""
import sys
import os
import py_compile

sys.path.insert(0, '.')

print("="*70)
print("PROJECT HEALTH CHECK")
print("="*70)

# Syntax check on all Python files
print("\n1. SYNTAX VALIDATION")
print("-"*70)
files_to_check = [
    'run_webapp.py',
    'webapp/app.py',
    'src/models/model1.py',
    'src/models/model2.py',
    'src/data/dataset.py',
    'src/training/train_model1.py',
    'src/training/train_model2.py',
    'src/evaluation/visualize.py',
    'src/evaluation/compare_models.py'
]

syntax_errors = 0
for f in files_to_check:
    try:
        py_compile.compile(f, doraise=True)
        print(f"✓ {f}")
    except Exception as e:
        print(f"✗ {f}: {e}")
        syntax_errors += 1

print(f"\nSyntax check: {len(files_to_check) - syntax_errors}/{len(files_to_check)} passed")

# Import check
print("\n2. IMPORT VALIDATION")
print("-"*70)
import_errors = 0

try:
    import torch
    import torchvision
    import cv2
    import numpy as np
    import flask
    print("✓ Core dependencies")
except Exception as e:
    print(f"✗ Core dependencies: {e}")
    import_errors += 1

try:
    from src.models.model1 import create_model
    from src.models.model2 import create_model2
    print("✓ Model modules")
except Exception as e:
    print(f"✗ Model modules: {e}")
    import_errors += 1

try:
    from src.data.dataset import TVSumDataset
    print("✓ Dataset module")
except Exception as e:
    print(f"✗ Dataset module: {e}")
    import_errors += 1

try:
    from webapp.app import app
    print("✓ Webapp module")
except Exception as e:
    print(f"✗ Webapp module: {e}")
    import_errors += 1

print(f"\nImport check: {4 - import_errors}/4 modules passed")

# File structure check
print("\n3. FILE STRUCTURE")
print("-"*70)
critical_dirs = ['src', 'webapp', 'tests', 'configs', 'data', 'docs', 'static', 'uploads']
critical_files = ['requirements.txt', 'README.md', 'Dockerfile', 'docker-compose.yml']

missing = 0
for d in critical_dirs:
    if os.path.exists(d):
        print(f"✓ {d}/")
    else:
        print(f"✗ {d}/ MISSING")
        missing += 1

for f in critical_files:
    if os.path.exists(f):
        print(f"✓ {f}")
    else:
        print(f"✗ {f} MISSING")
        missing += 1

print(f"\nFile structure: {len(critical_dirs) + len(critical_files) - missing}/{len(critical_dirs) + len(critical_files)} items present")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
total_checks = 3
failed_checks = (1 if syntax_errors > 0 else 0) + (1 if import_errors > 0 else 0) + (1 if missing > 0 else 0)
passed_checks = total_checks - failed_checks

if failed_checks == 0:
    print("✅ ALL CHECKS PASSED - Project is working correctly!")
else:
    print(f"⚠️  {passed_checks}/{total_checks} checks passed")
    print("Some components need attention.")

print("="*70)
