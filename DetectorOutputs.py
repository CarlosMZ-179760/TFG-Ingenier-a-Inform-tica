# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:29:04 2026

@author: TESTER
"""
import gc
from multiprocessing.pool import Pool
import pandas as pd
from capymoa.drift.eval_detector import EvaluateDriftDetector
from dataclasses import dataclass, asdict
import datetime
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
    ("ADWIN",{}),
    ("CUSUM",{}),
    ("DDM",{}),
    #"EWMAChart",
    ("GeometricMovingAverage",{}),
    ("HDDMAverage",{}),
    ("HDDMWeighted",{}),
    #"OPTWIN",
    #"PageHinkley",
    #"RDDM",
    #"SEED",
    #"STEPD",
    #"STUDD",
]

delay=500
detectorSet=[]
for detector_name, detector_args in all_detectors:
    detectorSet.append((getattr(detectors, detector_name)(**detector_args), detector_name))
reps=1#int(sys.argv[2])
start=1
drifts=1
seeds=np.arange(start,reps+start) #Número de semillas con el que se ejecutan las pruebas
low_error_rate=np.array([0.01
                         #,0.1
                         ,0.2
                         ])
magnitude_of_change=np.array([0.05
                              #,0.15
                              ,0.3
                              ])
stable_concept_duration=np.array([500
                                  #,10000
                                  ,100000
                                  ])
duration_of_change=np.array([0
                             #,100
                             ,1000
                             ])
noise_stable_concept=np.array([0
                               ,0.1
                               ])


folderpath = Path("Respuestas/StreamsSmallAlt")
folderpath.mkdir(parents=True, exist_ok=True)
def create_ensemble_log(stream):
    #sleep(100)
    #print("Hello", flush=True)
    
    if delay==0:
        valid_delay=stream[4]+stream[5]
    else:
        valid_delay=delay
    stream_duration=(2*(drifts+1)*(stream[4]+stream[5]))-stream[5]+valid_delay
    stream_name="Seed="+str(stream[0])+"; LER="+str(stream[1])+"; MoC="+str(stream[2])+"; NSC="+str(stream[3])+"; SCD="+str(stream[4])+"; DoC="+str(stream[5])
    filepath=folderpath.joinpath(str(stream_name)+ ".csv")
    print(stream_name, stream_duration)
    #print("File missing")
    now=datetime.datetime.now()
    print(str(now))
    trues=np.zeros(stream_duration+1000001)
    maxsDict={f"{detectorName}": [0,valid_delay] for detector, detectorName in detectorSet}
    maxsTrue=[0, valid_delay]
    resultsDict={f"{detectorName}": np.zeros(stream_duration+1000001) for detector, detectorName in detectorSet}
    #stream=stream_head+stream_tail
    #print(stream)
    
    
    generator=streamGen.GenericChangeGenerator(number_of_drifts=drifts, instance_random_seed=stream[0], low_error_level=stream[1], incr_error_level=stream[2], noise_stable_concept=stream[3], duration_stable_concept=stream[4], duration_change=stream[5], valid_delay=delay)
    i=0
    
    while generator.has_more_instances():
        stream_elem=generator.next_instance().x
        y=stream_elem[1]
        #print("y=", y)
        trues[i]=y
        #print(trues)
        #print(i)
        if(maxsTrue[1]<=0 or maxsTrue[0]<=y):
             maxsTrue[0]=y
             maxsTrue[1]=valid_delay
        maxsTrue[1]-=1
        trues[i]=maxsTrue[0]
        for detector, detectorName in detectorSet:
            #print(detector, detectorName)
            detector.add_element(stream_elem[0])
            if(detector.detected_warning()):
                last=1
                resultsDict[detectorName][i]=1
            elif(detector.detected_change()):
                #print("Change detected by: ",detectorName)
                last=2
                resultsDict[detectorName][i]=2
            else:
                last=0
                resultsDict[detectorName][i]=0
            if(maxsDict[detectorName][1]<=0 or maxsDict[detectorName][0]<last):
                maxsDict[detectorName][0]=last
                maxsDict[detectorName][1]=valid_delay
            maxsDict[detectorName][1]-=1
            resultsDict[detectorName][i]=maxsDict[detectorName][0]
            
        i+=1
    print(i)
    # print("Lectura de stream finalizada")
    # now=datetime.datetime.now()
    # print(str(now))
    # for detector, detectorName in detectorSet:
    #     resultsDict[detectorName] = np.array([max(resultsDict[detectorName][j-min(j,valid_delay):j+1]) for j in range(len(resultsDict[detectorName]))])
    # trues=np.array([max(trues[j-min(j,valid_delay):j+1]) for j in range(len(trues))])
    # print("Cálculo de máximo en ventana concluido")
    # now=datetime.datetime.now()
    # print(str(now))
    instance_eval=pd.DataFrame.from_dict(data=resultsDict)
    instance_eval["Ground-truth"]=trues
    # print("Dataframe creado")
    # now=datetime.datetime.now()
    # print(str(now))
    instance_eval.astype("int8")
    diff = (instance_eval.values[:-1]  == instance_eval.values[1:])
    repeated = np.insert(np.all(diff, axis=1), 0, True)
    repeated=np.reshape(repeated, (np.size(repeated),1))
    repeated=np.repeat(repeated, 7, axis=1)
    #print("Máscara creada")
    # # now=datetime.datetime.now()
    # # print(str(now))
    purged_eval=instance_eval.mask(repeated).dropna()
    # # print("Máscara aplicada")
    # # now=datetime.datetime.now()
    # # print(str(now))
    
    # #print(filepath)
    purged_eval.to_csv(filepath)
    # print("Registro guardado")
    # now=datetime.datetime.now()
    # print(str(now)+"\n")
    # del purged_eval
    # del filepath
    # del repeated
    # del diff
    # del instance_eval
    # del stream_duration
    # del stream_name
    # del now
    # del trues
    # del maxsDict
    # del maxsTrue
    # del resultsDict
    # del generator
    # del i
    # del y
    # del stream_elem
    for detector, _ in detectorSet:
        detector.reset(clean_history=True)
    return "OK"

# if __name__=='__main__':
#     print("true")
#     with Pool(8) as pool:
#         print(pool)
#         k=pool.starmap_async(streaming.create_ensemble_log, itertools.product(seeds, low_error_rate, magnitude_of_change, noise_stable_concept, stable_concept_duration, duration_of_change))
#         print(k)

# gc.collect(generation=0)
# gc.collect(generation=1)
# gc.collect(generation=2)
for stream_head in itertools.product(seeds, low_error_rate, magnitude_of_change, noise_stable_concept):
     for stream_tail in itertools.product(stable_concept_duration, duration_of_change):
         stream=stream_head+stream_tail
         create_ensemble_log(stream)

     gc.collect(generation=0)
     gc.collect(generation=1)
     gc.collect(generation=2)
        # trues=np.array([])
        # resultsDict={f"{detectorName}": np.array([]) for detector, detectorName in detectorSet}
        # stream=stream_head+stream_tail
        # #print(stream)
        # stream_name="Seed="+str(stream[0])+"; LER="+str(stream[1])+"; MoC="+str(stream[2])+"; NSC="+str(stream[3])+"; SCD="+str(stream[4])+"; DoC="+str(stream[5])
        # print(stream_name)
        
        # now = datetime.datetime.now()
        # print(str(now))
        # stream_duration=2*drifts*stream_tail[0]+2*drifts*stream_tail[1]+valid_delay
        # generator=streamGen.GenericChangeGenerator(instance_random_seed=stream[0], low_error_level=stream[1], incr_error_level=stream[2], noise_stable_concept=stream[3], duration_stable_concept=stream[4], duration_change=stream[5])
        # i=0
        
        # while generator.has_more_instances() and i<stream_duration:
        #     stream_elem=generator.next_instance().x
        #     y=stream_elem[1]
        #     #print("y=", y)
        #     trues=np.append(trues,[y])
        #     #print(trues)
        #     #print(i)
        #     for detector, detectorName in detectorSet:
        #         #print(detector, detectorName)
        #         detector.add_element(stream_elem[0])
        #         if(detector.detected_warning()):
        #             resultsDict[detectorName]=np.append(resultsDict[detectorName],[1])
        #         elif(detector.detected_change()):
        #             #print("Change detected by: ",detectorName)
        #             resultsDict[detectorName]=np.append(resultsDict[detectorName],[2])
        #         else:
        #             resultsDict[detectorName]=np.append(resultsDict[detectorName],[0]) 
        #     i+=1
        # for detector, detectorName in detectorSet:
        #     resultsDict[detectorName] = np.array([max(resultsDict[detectorName][j-min(j,valid_delay):j+1]) for j in range(len(resultsDict[detectorName]))])
        # instance_eval=pd.DataFrame.from_dict(data=resultsDict)
        # trues=np.array([max(trues[j-min(j,valid_delay):j+1]) for j in range(len(trues))])
        # instance_eval["Ground-truth"]=trues
        # instance_eval.astype("int32")
        # diff = (instance_eval.values[:-1]  == instance_eval.values[1:])
        # repeated = np.insert(np.all(diff, axis=1), 0, True)
        # repeated=np.reshape(repeated, (np.size(repeated),1))
        # repeated=np.repeat(repeated, 7, axis=1)
        # purged_eval=instance_eval.mask(repeated).dropna()
        # filepath=folderpath.joinpath(str(stream_name)+ ".csv")
        # print(filepath)
        # purged_eval.to_csv(filepath)
        # for detector, detectorName in detectorSet:
        #     detector.reset() #Si se quiere no reiniciar los detectores tras cada stream, se pueden comentar estas dos líneas
        # for detectorName in all_detectors: 
        #     detector = getattr(detectors, detectorName)()
        #     #print(detector)
        #     detected_drifts=0
        #     #print(stream[0],stream[4], stream[5], 2*drifts*stream[4]+2*drifts*stream[5]+valid_delay)
        #     drift_eval = EvaluateDriftDetector(max_delay=valid_delay)
            
        #     i=0
        #     while generator.has_more_instances() and detector.idx<=stream_duration:
        #         stream_elem=generator.next_instance().x
        #         y=stream_elem[1]
        #         #print("y=", y)
        #         trues=np.append(trues,[y])
        #         #print(trues)
        #         detector.add_element(stream_elem[0])
        #         if(detector.detected_warning()):
        #             resultsDict[detectorName]=np.append(resultsDict[detectorName],[1])
        #         elif(detector.detected_change()):
        #             #print("Change detected by: ",detectorName)
        #             resultsDict[detectorName]=np.append(resultsDict[detectorName],[2])
        #         else:
        #             resultsDict[detectorName]=np.append(resultsDict[detectorName],[0])            
            #print(resultsDict)
# for detector, detectorName in detectorSet:
#     resultsDict[detectorName] = np.array([max(resultsDict[detectorName][j-min(j,valid_delay):j+1]) for j in range(len(resultsDict[detectorName]))])
#         #print(resultsDict)
# instance_eval=pd.DataFrame.from_dict(data=resultsDict)
# trues=np.array([max(trues[j-min(j,valid_delay):j+1]) for j in range(len(trues))])
# instance_eval["Ground-truth"]=trues
# instance_eval.astype("int32")
# diff = (instance_eval.values[:-1]  == instance_eval.values[1:])
# repeated = np.insert(np.all(diff, axis=1), 0, True)
# repeated=np.reshape(repeated, (np.size(repeated),1))
# repeated=np.repeat(repeated, 7, axis=1)
# purged_eval=instance_eval.mask(repeated).dropna()
# filepath=folderpath.joinpath("test-multistream-max"+ ".csv")
# purged_eval.to_csv(filepath)