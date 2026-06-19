# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:48:48 2026

@author: TESTER
"""

import capymoa.drift.detectors.ensemble_detector as ensemble
import capymoa.stream.generator as streamGen
from pathlib import Path
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
repsStr=3
startSeed=int(seedStr)
reps=int(repsStr)
driftList=[10,20,30,40]#np.arange(20, 30,10)
seeds=np.arange(startSeed,reps+startSeed) #Número de semillas con el que se ejecutan las pruebas
low_error_rate=np.array([0.01,
                         0.2])
errorRateIndex=sys.argv[2]
selectedErrorRate=np.array([low_error_rate[int(errorRateIndex)]])
#MOCIndex=sys.argv[2]
magnitude_of_change=np.array([0.3,
                              0.05])
selectedMOC=magnitude_of_change#np.array([magnitude_of_change[int(MOCIndex)]])
#NSCIndex=sys.argv[3]
noise_stable_concept=np.array([0
                               ,0.1
                               ])
selectedNSC=noise_stable_concept#np.array([noise_stable_concept[int(NSCIndex)]])



stable_concept_duration=np.array([5000
                                  ,100000
                                  ])
duration_of_change=np.array([1000,
                             0])
valid_delays=np.array([500])

resultspath = Path("Resultados/Ensemble/Rendimiento")
resultspath.mkdir(parents=True, exist_ok=True)
dictspath = Path("Resultados/Ensemble/Diccionarios")
dictspath.mkdir(parents=True, exist_ok=True)
freqspath = Path("Resultados/Ensemble/Frecuencias")
freqspath.mkdir(parents=True, exist_ok=True)
logspath = Path("Resultados/Ensemble/Logs")
logspath.mkdir(parents=True, exist_ok=True)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('Resultados/Ensemble/output-EnsembleExperimentLog'
                        +sys.argv[1]
                        +"-"
                        +sys.argv[2]
                        +'-Oza(NB).txt','wt')
   
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


#print("Inicio", flush=True)
#for detectorName in all_detectors:
#frames=[]
#dicts=[]
#freqs=[]
largeIterCounter=0
fileNumber=0
start=datetime.datetime.now()
end=start
print("Hora de inicio", start)
largeIters=len(seeds)*len(selectedErrorRate)*len(selectedMOC)*len(selectedNSC)
for stream_head in itertools.product(seeds,selectedErrorRate, selectedMOC, selectedNSC):
    largeIterCounter+=1
    #print(k,j)#, flush=True)
    for stream_tail in itertools.product(stable_concept_duration, duration_of_change):
        # detector = ensemble.EnsembleDetector(validDelay=valid_delay)
        # runningDrifts=0
        # streamProgress=0
        # stream=stream_head+stream_tail
        # maxDrifts=max(driftList)
        # apparentDelay=stream[5]+valid_delay
        # datasetGenerator=streamGen.GenericChangeGenerator(validDelay=apparentDelay, number_of_drifts=maxDrifts, duration_change=stream[5], instance_random_seed=stream[0], duration_stable_concept=stream[4])
        
        for valid_delay in valid_delays:
            streamProgress=0
            stream=stream_head+stream_tail

            maxDrifts=max(driftList)
            apparentDelay=stream[5]+valid_delay
            detector = ensemble.EnsembleDetector(valid_delay=apparentDelay)#, detectorDict=[("ADWIN",{}),
            #               ("CUSUM",{}),
            #               ("DDM",{}),
            #               ("EDDM",{}),
            #               ("HDDMAverage",{}),
            #               ("HDDMWeighted",{})]
            #)
            logCols=tuple(detector.baseDetectorNameList)+tuple(["Prediction","Ground-Truth"])
            #runningDrifts=0
            
            datasetGenerator=streamGen.GenericChangeGenerator(noise_stable_concept=stream[3],noise_change=stream[3],valid_delay=apparentDelay, number_of_drifts=maxDrifts, duration_change=stream[5], instance_random_seed=stream[0], duration_stable_concept=stream[4])
            for drifts in driftList:
                dfIndex=stream+tuple([drifts,valid_delay])
                fileNumber+=1
                stream_duration=2*drifts*stream_tail[0]+2*drifts*stream_tail[1]+valid_delay
                filepath=resultspath.joinpath("Perf( Seed="+str(stream[0])+";LER="+str(stream[1])+";MOC="+str(stream[2])+";NSC="+str(stream[3])+";SCD="+str(stream[4])+";DOC="+str(stream[5])+";NDs="+str(drifts)+";VD="+str(valid_delay)+").csv")
                dictpath=dictspath.joinpath("Dict( Seed="+str(stream[0])+";LER="+str(stream[1])+";MOC="+str(stream[2])+";NSC="+str(stream[3])+";SCD="+str(stream[4])+";DOC="+str(stream[5])+";NDs="+str(drifts)+";VD="+str(valid_delay)+").csv")
                freqpath=freqspath.joinpath("Freqs( Seed="+str(stream[0])+";LER="+str(stream[1])+";MOC="+str(stream[2])+";NSC="+str(stream[3])+";SCD="+str(stream[4])+";DOC="+str(stream[5])+";NDs="+str(drifts)+";VD="+str(valid_delay)+").csv")
                logpath=logspath.joinpath("Logs( Seed="+str(stream[0])+";LER="+str(stream[1])+";MOC="+str(stream[2])+";NSC="+str(stream[3])+";SCD="+str(stream[4])+";DOC="+str(stream[5])+";NDs="+str(drifts)+";VD="+str(valid_delay)+").csv")
                fileExists=filepath.is_file() and dictpath.is_file() and freqpath.is_file()
                
                streamDuration=2*(drifts)*(stream[4]+stream[5])-stream[5]+apparentDelay
                datasetSize=streamDuration-streamProgress
                print("Semilla:",stream[0],"\n Tasa de error baja:",stream[1],"\n Aumento de la tasa de error:",stream[2],"\n Ruido del concepto estable:", stream[3], "\n Duración del concepto estable:",stream[4],"\n Duración del cambio:",stream[5],"\n Número de cambios:",drifts,"\nRetreaso aceptable:",valid_delay)
                logs=np.empty(shape=[streamDuration, len(detector.baseDetectorNameList)+2])
                readings=np.zeros([datasetSize,1])
                driftHistory=np.zeros([datasetSize])
                #print("\n")
                #datasetGenerator=streamGen.GenericChangeGenerator(valid_delay=apparentDelay, number_of_drifts=drifts, duration_change=stream[5], instance_random_seed=stream[0], duration_stable_concept=stream[4])
                for i in range(datasetSize):
                    stream_elem=datasetGenerator.next_instance().x
                    atom=stream_elem[0]
                    y=stream_elem[1]
                    #print("y=", y)
                    readings[i]=atom
                    driftHistory[i]=y
                print("Nuevos elementos de entrenamiento en la iteración",readings.shape)
                #i=0
                
                #, flush=True)
                #stream_duration=2*(drifts)*(stream[4]+stream[5])-stream[5]+apparentDelay
                #print(stream_duration)
                #print(readings, drifts)
                trainingStream : NumpyStream[LabeledInstance]=NumpyStream(readings, driftHistory, dataset_name='trainingStream', feature_names=['classifierError'], target_name='Drift', target_type='categorical')
                #detector = ensemble.EnsembleDetector(valid_delay=apparentDelay, datasetSize=streamDuration, dataset=trainingStream)
                detector.feed_training_data(datasetSize, trainingStream)
                detector.preprocess_training_data()
                detector.trainClassifier()
                #detector = ensemble.EnsembleDetector(validDelay=valid_delay)
                #retrain=True
                testGenerator=streamGen.GenericChangeGenerator(valid_delay=apparentDelay,instance_random_seed=stream[0]+reps, low_error_level=stream[1], incr_error_level=stream[2], noise_stable_concept=stream[3],noise_change=stream[3] ,duration_stable_concept=stream[4], duration_change=stream[5], number_of_drifts=drifts)
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
                j=0
                detector.deleteDetections()
                testStartTime=datetime.datetime.now()
                for instance in testGenerator:
                    prediction=0
                    stream_elem=instance.x
                    y=stream_elem[1]
                    detector.add_element(stream_elem[0])
                    if detector.detected_change():
                        #print("El detector informa de un cambio",j)
                        prediction=1
                    #else:
                    #    prediction=0
                    if (y==1):
                        #print(instance)
                        #print(j)
                        #print("y=", y)
                        trues.append(j)
                        #print(trues)
                    logs[j,:]=np.array([tuple(detector.maxsWindow[:,0])+tuple([prediction, y])],dtype=np.int8)
                    #print(logs[j,:],j)
                    #time.sleep(0.0001)
                    j+=1
                #j+=1
                preds=detector.detection_index
                print(trues, preds)
                #(dictionary.keys())
                freqsKeys=[tuple(feature+tuple([value])) for feature,value in itertools.product(dictionary.keys(),[0,1])]
                #print(freqsKeys)
                logDf=pd.DataFrame(data=logs, 
                                   columns=logCols, 
                                   dtype=np.int16)
                diff = (logDf.values[:-1]  == logDf.values[1:])
                repeated = np.insert(np.all(diff, axis=1), 0, True)
                repeated=np.reshape(repeated, (np.size(repeated),1))
                repeated=np.repeat(repeated, 7, axis=1)
                # #print("Máscara creada")
                # # # # now=datetime.datetime.now()
                # # # # print(str(now))
                purgedLogDf=logDf.mask(repeated).dropna()
                # # # # print("Máscara aplicada")
                # # # # now=datetime.datetime.now()
                # # # # print(str(now))
                
                # # # #print(filepath)
                purgedLogDf.to_csv(logpath)
                
                # #print(detector.resultsDictionary.keys())
                results=drift_eval.calc_performance(trues, 
                                                    preds, 
                                                    tot_n_instances=detector.idx)
                print(results)
                instance_eval=pd.DataFrame(data=asdict(results), 
                                           index=pd.MultiIndex.from_tuples([dfIndex], 
                                                                           names=("Seed", "Low error rate", "Magnitude of change", "Noise level", 
                                                                                  "Stable concept duration", "Duration of change", "Drifts", "Valid Delay")))
                dictEval=pd.DataFrame(data=dictionary,
                                      columns=dictionary.keys(),
                                           index=pd.MultiIndex.from_tuples([dfIndex], 
                                                                           names=("Seed", "Low error rate", "Magnitude of change", "Noise level", 
                                                                                  "Stable concept duration", "Duration of change", "Drifts", "Valid Delay")))
                freqsDFNan=pd.DataFrame(data=freqsDict,
                                      columns=freqsKeys,
                                           index=pd.MultiIndex.from_tuples([dfIndex], 
                                                                           names=("Seed", "Low error rate", "Magnitude of change", "Noise level", 
                                                                                  "Stable concept duration", "Duration of change", "Drifts", "Valid Delay")))
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
                del dictionary
                del freqsDict
                del readings
                del driftHistory
                #del datasetSize
                del stream_elem
                del atom
                del y
                #del datasetGenerator
                del testGenerator
                del trainingStream
                # del logs
                # del logDf
                # del diff
                # del repeated
                # del purgedLogDf
                runningDrifts=drifts
                gc.collect(generation=0)
                gc.collect(generation=1)
                gc.collect(generation=2)
                endold=end
                end=datetime.datetime.now()
                print("Duración del procesamiento de stream", end-endold, float(streamDuration)/(end-testStartTime).total_seconds())
                print("Duración de test:", end-testStartTime)
                print("Tiempo total transcurrido",end-start, flush=True)
                print("Hora del sistema", end)
                streamProgress=streamDuration
            #detectorDictionariesAvg=pd.concat(framesAvgs)
            #del datasetGenerator
            #del detector
            #del logCols
            gc.collect(generation=0)
            gc.collect(generation=1)
            gc.collect(generation=2)
        #print(end-start)
    j+=1
    print("Hora estimada de finalización",start+(end-start)*(largeIters/largeIterCounter), flush=True)
    
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