#!/usr/bin/env python3
"""Test script to verify sampler fix for classification datasets with small classes."""

import numpy as np
import sys

# Simulate Breast dataset class distribution
y = np.array([0]*28 + [1]*10 + [2]*149 + [3]*405 + [5]*91 + [7]*20)

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
min_class_size = counts.min()
n_classes = len(counts)

print("Breast Dataset Analysis:")
print(f"Class distribution: {dict(zip(unique, counts))}")
print(f"Total samples: {len(y)}")
print(f"Number of classes: {n_classes}")
print(f"Minimum class size: {min_class_size}")

# Test the conservative threshold logic
CV_INNER = 3
conservative_threshold = CV_INNER * 4  # 12

print(f"\nSampler Decision Logic:")
print(f"Conservative threshold (CV_INNER * 4): {conservative_threshold}")
print(f"Min class size >= threshold: {min_class_size >= conservative_threshold}")

# Check for too many small classes
small_classes = np.sum(counts < conservative_threshold)
too_many_small = small_classes > n_classes / 2

print(f"Small classes (< {conservative_threshold}): {small_classes}/{n_classes}")
print(f"Too many small classes: {too_many_small}")

# Final decision
if too_many_small:
    decision = "SKIP (too many small classes)"
elif min_class_size >= conservative_threshold:
    decision = "USE SAMPLER"
else:
    decision = "SKIP (min class too small)"

print(f"\nFinal Decision: {decision}")

# Test what happens in CV splits
print(f"\nCV Split Analysis:")
print(f"With {CV_INNER}-fold CV, each fold gets ~{len(y)//CV_INNER} samples")
print(f"Worst case for smallest class ({min_class_size} samples):")
print(f"  - Could have as few as {min_class_size//CV_INNER} samples per fold")
print(f"  - Some folds might have 0-1 samples for this class") 