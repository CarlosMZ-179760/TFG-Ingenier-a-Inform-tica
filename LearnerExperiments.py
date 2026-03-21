# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:21:53 2026

@author: TESTER
"""

import pandas as pd

import capymoa.drift.detectors as detectors
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools


#from moa.streams.generators.cd import GenericChangeGenerator as MOA_GenericChangeGenerator
import capymoa.drift.detectors.ensemble_detector as ensemble
import capymoa.stream.generator as streamGen

all_detectors = detectors.__all__
reps=10
drifts=10
seeds=np.arange(1,reps+1) #Número de semillas con el que se ejecutan las pruebas
low_error_rate=np.array([0.01, 0.2])
magnitude_of_change=np.array([0.05, 0.3])
stable_concept_duration=np.array([500,100000])
duration_of_change=np.array([0,1000])

#stream_params = itertools.product(seeds, low_error_rate, magnitude_of_change, stable_concept_duration, duration_of_change):
    
for (detectorName, stream) in itertools.product(all_detectors, itertools.product(seeds, low_error_rate, magnitude_of_change, stable_concept_duration, duration_of_change)):
    generator=streamGen.GenericChangeGenerator(instance_random_seed=stream[0], low_error_level=stream[1], incr_error_level=stream[2], duration_stable_concept=stream[3], duration_change=stream[4])
    detector = getattr(detectors, detectorName)()
    print(detector)
    detected_drifts=0
    print(stream[3], stream[4], 2*drifts*stream[3]+(2*drifts+1)*stream[4])
    while generator.has_more_instances() and detector.idx<=(2*drifts+1)*stream[3]+2*(drifts+1)*stream[4]:
        if (detector.idx>9000):
            print(detector.idx, detected_drifts)
        i=generator.next_instance().x
        detector.add_element(i[0])
        detected_drifts+=i[1]