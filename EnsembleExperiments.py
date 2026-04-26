# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:48:48 2026

@author: TESTER
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from capymoa.drift.eval_detector import EvaluateDriftDetector
from dataclasses import dataclass, asdict
import datetime
import time
from capymoa.stream import NumpyStream
from capymoa.instance import Instance, LabeledInstance, RegressionInstance
import capymoa.drift.detectors as detectors
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import os
import gc


#from moa.streams.generators.cd import GenericChangeGenerator as MOA_GenericChangeGenerator
seedStr=sys.argv[1]
repsStr=sys.argv[2]
startSeed=int(seedStr)
reps=int(repsStr)
driftList=[20,30,40]#np.arange(20, 30,10)
seeds=np.arange(startSeed,reps+startSeed) #Número de semillas con el que se ejecutan las pruebas
low_error_rate=np.array([0.01, 
                         0.2])
magnitude_of_change=np.array([0.05, 
                              0.3])
stable_concept_duration=np.array([5000
                                  ,100000
                                  ])
duration_of_change=np.array([0,
                             1000])
noise_stable_concept=np.array([0
                               ,0.1
                               ])
valid_delay=500

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('output-EnsembleExperimentLog'+seedStr+"-"+repsStr+'-Oza(NB).txt','w')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        #self.write(self.terminal.buffer)
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

import capymoa.drift.detectors.ensemble_detector as ensemble
import capymoa.stream.generator as streamGen
from pathlib import Path
#sys.stdout = open('output-EnsembleExperimentLog'+str(start)+'-Oza(NB).txt','w')
#sys.stdout=sys.__stdout__
all_detectors = [
    # "ABCD",
    "ADWIN",
    "CUSUM",
    "DDM",
    #"EWMAChart",
    #"GeometricMovingAverage",
    "HDDMAverage",
    "HDDMWeighted",
    #"OPTWIN",
    #"PageHinkley",
    #"RDDM",
    #"SEED",
    #"STEPD",
    #"STUDD",
]

resultspath = Path("Resultados/Ensemble/Rendimiento")
resultspath.mkdir(parents=True, exist_ok=True)
dictspath = Path("Resultados/Ensemble/Diccionarios")
dictspath.mkdir(parents=True, exist_ok=True)
freqspath = Path("Resultados/Ensemble/Frecuencias")
freqspath.mkdir(parents=True, exist_ok=True)
#print("Inicio", flush=True)
#for detectorName in all_detectors:
#frames=[]
#dicts=[]
#freqs=[]
j=0
k=0
start=datetime.datetime.now()
end=start
print("Hora de inicio", start)
for stream_head in itertools.product(seeds,low_error_rate, magnitude_of_change, noise_stable_concept):
    print(k,j)#, flush=True)
    for stream_tail in itertools.product(stable_concept_duration, duration_of_change):
        for drifts in driftList:
        #stream_duration=2*drifts*stream_tail[0]+2*drifts*stream_tail[1]+valid_delay
                filepath=resultspath.joinpath("Perf ("+seedStr+"-"+repsStr+"-"+str(k)+").csv")
                dictpath=dictspath.joinpath("Dicts ("+seedStr+"-"+repsStr+"-"+str(k)+").csv")
                freqpath=freqspath.joinpath("Freqs ("+seedStr+"-"+repsStr+"-"+str(k)+").csv")
            #fileExists=filepath.is_file() and dictpath.is_file() and freqpath.is_file()
            #if (fileExists):
                k+=1
                stream=stream_head+stream_tail+tuple([drifts])
                drifts=stream[6]
                print("Semilla:",stream[0],"\n Tasa de error baja:",stream[1],"\n Aumento de la tasa de error:",stream[2],"\n Ruido del concepto estable:", stream[3], "\n Duración del concepto estable:",stream[4],"\n Duración del cambio:",stream[5],"\n Número de cambios:",drifts)
                apparentDelay=stream[5]+valid_delay
                readings=np.zeros([2*(drifts+1)*(stream[4]+stream[5])-stream[5]+apparentDelay,1])
                driftHistory=np.zeros([2*(drifts+1)*(stream[4]+stream[5])-stream[5]+apparentDelay])
                #print("\n")
                datasetGenerator=streamGen.GenericChangeGenerator(valid_delay=apparentDelay, number_of_drifts=drifts, duration_change=stream[5], instance_random_seed=stream[0], duration_stable_concept=stream[4])
                i=0
                while datasetGenerator.has_more_instances():
                    stream_elem=datasetGenerator.next_instance().x
                    atom=stream_elem[0]
                    y=stream_elem[1]
                    #print("y=", y)
                    readings[i]=atom
                    driftHistory[i]=y
                    i+=1
                #print(i)
                #, flush=True)
                stream_duration=2*(drifts+1)*(stream[4]+stream[5])-stream[5]+apparentDelay
                print(stream_duration)
                #print(readings, drifts)
                trainingStream : NumpyStream[LabeledInstance]=NumpyStream(readings, driftHistory, dataset_name='trainingStream', feature_names=['classifierError'], target_name='Drift', target_type='categorical')
                detector = ensemble.EnsembleDetector(dataset=trainingStream, valid_delay=apparentDelay, datasetSize=np.size(driftHistory))
                testGenerator=streamGen.GenericChangeGenerator(instance_random_seed=stream[0]+reps, low_error_level=stream[1], incr_error_level=stream[2], noise_stable_concept=stream[3], duration_stable_concept=stream[4], duration_change=stream[5])
                #detector = detectors.ensemble_detector.EnsembleDetector(dataset=trainingStream, valid_delay=apparentDelay, datasetSize=2*(drifts+1)*(stream[4]+stream[5])-stream[5]+apparentDelay)
                #detector.pretrain()
                #detector.describe_drifts()
                dictionary=detector.resultsDictionary
                #print(dictionary)
                freqsDict=detector.frequencies
                #print(freqsDict)
                #break
                detected_drifts=0
                #print(stream[0],stream[3], stream[4], 2*drifts*stream[3]+(2*drifts+1)*stream[4])
                #print(apparentDelay)
                drift_eval = EvaluateDriftDetector(max_delay=int(apparentDelay))
                trues=[]
                # i=0
                for instance in testGenerator:
                    stream_elem=instance.x
                    y=stream_elem[1]
                    if (y==1):
                        #print("y=", y)
                        trues.append(i)
                        #print(trues)
                    detector.add_element(stream_elem[0])
                    #i+=1
                #j+=1
                preds=detector.detection_index
                #print(trues, preds)
                #print(detector.resultsDictionary.keys())
                results=drift_eval.calc_performance(trues, 
                                                    preds, 
                                                    tot_n_instances=detector.idx)
                instance_eval=pd.DataFrame(data=asdict(results), 
                                           index=pd.MultiIndex.from_tuples([stream], 
                                                                           names=("Seed", "Low error rate", "Magnitude of change", "Noise level", 
                                                                                  "Stable concept duration", "Duration of change", "Drifts")))
                dictEval=pd.DataFrame(data=dictionary,
                                      columns=dictionary.keys(),
                                           index=pd.MultiIndex.from_tuples([stream], 
                                                                           names=("Seed", "Low error rate", "Magnitude of change", "Noise level", 
                                                                                  "Stable concept duration", "Duration of change", "Drifts")))
                freqsDFNan=pd.DataFrame(data=freqsDict,
                                      columns=dictionary.keys(),
                                           index=pd.MultiIndex.from_tuples([stream], 
                                                                           names=("Seed", "Low error rate", "Magnitude of change", "Noise level", 
                                                                                  "Stable concept duration", "Duration of change", "Drifts")))
                freqsDF=freqsDFNan.fillna(0)
                del freqsDFNan
                #frames.append(instance_eval)
                #print(freqsDF)
                #dicts.append(dictEval)
                #freqs.append(freqsDF)
                #framesAvgs.append(instance_eval)
                #dictsAvgs.append(dictEval)
                instance_eval.to_csv(filepath)
                dictEval.to_csv(dictpath)
                freqsDF.to_csv(freqpath)
                del instance_eval
                del dictEval
                del freqsDF
                del detector
                del dictionary
                del freqsDict
                del readings
                del driftHistory
                del datasetGenerator
                del testGenerator
                del trainingStream
                gc.collect(generation=0)
                gc.collect(generation=1)
                gc.collect(generation=2)
                endold=end
                end=datetime.datetime.now()
                print("Duración del procesamiento de stream", end-endold, float(stream_duration)/(end-endold).total_seconds())
                print("Tiempo total transcurrido",end-start, flush=True)
                print("Hora del sistema", end)
            #detectorDictionariesAvg=pd.concat(framesAvgs)
                
        #print(end-start)
    j+=1
    print("Hora estimada de finalización",start+(end-start)*(8/j), flush=True)
    
# detectorResults=pd.concat(frames)
# detectorDictionaries=pd.concat(dicts)
# detectorFrequencies=pd.concat(freqs)
# #detectorResultsAvg=detectorResults.groupby(["Seed"]).mean()
# #detectorDictionariesAvg=detectorDictionaries.groupby(["Seed"]).mean()
# #detectorDictionariesVar=detectorDictionariesAvg-detectorDictionariesAvg.pow(2)
# filepath=folderpath.joinpath("Rendimiento/Ensemble-Large ("+start+"-"+reps+").csv")
# dictpath=folderpath.joinpath("Diccionarios/Dicts-Large ("+start+"-"+reps+").csv")
# freqpath=folderpath.joinpath("Diccionarios/Freqs-Large ("+start+"-"+reps+").csv")
# detectorResults.to_csv(filepath)
# detectorDictionaries.to_csv(dictpath)
# detectorFrequencies.to_csv(freqpath)