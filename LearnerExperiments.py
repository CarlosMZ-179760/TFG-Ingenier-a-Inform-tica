# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:21:53 2026

@author: TESTER
"""

import pandas as pd
from capymoa.drift.eval_detector import EvaluateDriftDetector
from dataclasses import dataclass, asdict

import time
import capymoa.drift.detectors as detectors
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import os

#from moa.streams.generators.cd import GenericChangeGenerator as MOA_GenericChangeGenerator
import capymoa.drift.detectors.ensemble_detector as ensemble
import capymoa.stream.generator as streamGen
from pathlib import Path
all_detectors = [
    #"ABCD",
    "ADWIN",
    "CUSUM",
    "DDM",
    #"EWMAChart",
    "GeometricMovingAverage",
    "HDDMAverage",
    "HDDMWeighted",
    #"OPTWIN",
    #"PageHinkley",
    #"RDDM",
    #"SEED",
    #"STEPD",
    #"STUDD",
]
reps=15
drifts=10
seeds=np.arange(1+15,reps+15+1) #Número de semillas con el que se ejecutan las pruebas
low_error_rate=np.array([0.01, 0.2])
magnitude_of_change=np.array([0.05, 0.3])
stable_concept_duration=np.array([500,100000])
duration_of_change=np.array([0,1000])
noise_stable_concept=np.array([0,0.1])
valid_delay=500

folderpath = Path("Resultados")
folderpath.mkdir(parents=True, exist_ok=True)
k=1
for detectorName in all_detectors:
    frames=[]
    j=0
    for stream_head in itertools.product(low_error_rate, magnitude_of_change, noise_stable_concept):
        print(k,j)
        for stream_tail in itertools.product(stable_concept_duration, duration_of_change):
            for seed in seeds:
                stream_duration=2*drifts*stream_tail[0]+2*drifts*stream_tail[1]+valid_delay
                stream=tuple([seed])+stream_head+stream_tail
                generator=streamGen.GenericChangeGenerator(instance_random_seed=stream[0], low_error_level=stream[1], incr_error_level=stream[2], noise_stable_concept=stream[3], duration_stable_concept=stream[4], duration_change=stream[5])
                detector = getattr(detectors, detectorName)()
                #print(detector)
                detected_drifts=0
                #print(stream[0],stream[3], stream[4], 2*drifts*stream[3]+(2*drifts+1)*stream[4])
                drift_eval = EvaluateDriftDetector(max_delay=valid_delay)
                trues=[]
                i=0
                while generator.has_more_instances() and detector.idx<=stream_duration:
                    stream_elem=generator.next_instance().x
                    y=stream_elem[1]
                    if (y==1):
                        #print("y=", y)
                        trues.append(i)
                        #print(trues)
                    detector.add_element(stream_elem[0])
                    i+=1
                preds=detector.detection_index
                #print(trues, preds)
                results=drift_eval.calc_performance(trues, 
                                                    preds, 
                                                    tot_n_instances=detector.idx)
                instance_eval=pd.DataFrame(data=asdict(results), 
                                           index=pd.MultiIndex.from_tuples([stream], 
                                                                           names=("Seed", "Low error rate", "Magnitude of change", "Noise level", 
                                                                                  "Stable concept duration", "Duration of change")))
                frames.append(instance_eval)
            
        j+=1
    k+=1
    detectorResults=pd.concat(frames)
    filepath=folderpath.joinpath(detectorName+ ".csv")
    detectorResults.to_csv(filepath)

    