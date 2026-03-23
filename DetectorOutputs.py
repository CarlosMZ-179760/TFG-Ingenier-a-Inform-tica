# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:29:04 2026

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
reps=1
drifts=1
seeds=np.arange(1,reps+1) #Número de semillas con el que se ejecutan las pruebas
low_error_rate=np.array([#0.01, 
                         0.2])
magnitude_of_change=np.array([#0.05, 
                              0.3])
stable_concept_duration=np.array([#500,
                                  100000])
duration_of_change=np.array([#0,
                             1000])
noise_stable_concept=np.array([#0,
                               0.1])
valid_delay=500

folderpath = Path("Respuestas")
folderpath.mkdir(parents=True, exist_ok=True)

for stream_head in itertools.product(seeds, low_error_rate, magnitude_of_change, noise_stable_concept):
    for stream_tail in itertools.product(stable_concept_duration, duration_of_change):
        stream=stream_head+stream_tail
        resultsDict={f"{detectorName}": np.array([]) for detectorName in all_detectors}
        for detectorName in all_detectors:
            generator=streamGen.GenericChangeGenerator(instance_random_seed=stream[0], low_error_level=stream[1], incr_error_level=stream[2], noise_stable_concept=stream[3], duration_stable_concept=stream[4], duration_change=stream[5])
            detector = getattr(detectors, detectorName)()
            #print(detector)
            detected_drifts=0
            #print(stream[0],stream[3], stream[4], 2*drifts*stream[3]+(2*drifts+1)*stream[4])
            drift_eval = EvaluateDriftDetector(max_delay=valid_delay)
            trues=np.array([])
            while generator.has_more_instances() and detector.idx<=(2*drifts+1)*stream[3]+2*(drifts+1)*stream[4]:
                stream_elem=generator.next_instance().x
                y=stream_elem[1]
                #print("y=", y)
                trues=np.append(trues,[y])
                #print(trues)
                detector.add_element(stream_elem[0])
                if(detector.detected_warning()):
                    resultsDict[detectorName]=np.append(resultsDict[detectorName],[1])
                elif(detector.detected_change()):
                    resultsDict[detectorName]=np.append(resultsDict[detectorName],[2])
                else:
                    resultsDict[detectorName]=np.append(resultsDict[detectorName],[0])            
            #print(resultsDict)
            resultsDict[detectorName] = np.array([max(resultsDict[detectorName][j-min(j,valid_delay):j+1]) for j in range(len(resultsDict[detectorName]))])
        #print(resultsDict)
        instance_eval=pd.DataFrame.from_dict(data=resultsDict)
        trues=np.array([max(trues[j-min(j,valid_delay):j+1]) for j in range(len(trues))])
        instance_eval["Ground-truth"]=trues
        instance_eval.astype("int32")
        diff = (instance_eval.values[:-1]  == instance_eval.values[1:])

# [[ True,  True,  True],
#  [ True,  True,  True],
#  [False,  True,  True],
#  [False,  True,  True],
#  [False,  True,  True],
#  [ True,  True,  True]]

# collapse rows into single value np.all(..., axis=1)
# make array len == number of rows in original DF
        #print(diff, np.shape(diff))
        repeated = np.insert(np.all(diff, axis=1), 0, True)
        repeated=np.reshape(repeated, (4001,1))
        repeated=np.repeat(repeated, 7, axis=1)
        #print(np.transpose(repeated[121:130]))
# [False,  True,  True, False, False, False,  True]

# modify df in-place
        #instance_eval.values[repeated] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        #print(np.shape(repeated), np.shape(instance_eval.values))
        #print(instance_eval.values[121:130])
        purged_eval=instance_eval.mask(repeated).dropna()
        #print(purged_eval)
        # filterdups=(instance_eval["ADWIN"]!=instance_eval["ADWIN"].shift() and 
        #         instance_eval["CUSUM"]!=instance_eval["CUSUM"].shift() and 
        #         instance_eval["DDM"]!=instance_eval["DDM"].shift() and
        #         instance_eval["GeometricMovingAverage"]!=instance_eval["GeometricMovingAverage"].shift() and 
        #         instance_eval["HDDMAverage"]!=instance_eval["HDDMAverage"].shift() and
        #         instance_eval["HDDMWeighted"]!=instance_eval["HDDMWeighted"].shift() and
        #         instance_eval["Ground-truth"]!=instance_eval["Ground-truth"].shift())
        #purged_eval=instance_eval[filterdups]
        filepath=folderpath.joinpath("results"+ ".csv")
        purged_eval.to_csv(filepath)