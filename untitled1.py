# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:22:08 2026

Borrador para implementar la toma de datos sobre los detectores base a usar en un ensemble.

@author: TESTER
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict


import capymoa.drift.detectors.ensemble_detector as ensemble

detectorDict=[("HDDMAverage", {}),("CUSUM", {}),("ADWIN", {}), ("PageHinkley",{}), ("STEPD",{})]

detector = ensemble.EnsembleDetector(detectorDict)

base_detectors=detector.get_params().get("baseDetectorsList")
for d in base_detectors:
    print(d.get_params())
