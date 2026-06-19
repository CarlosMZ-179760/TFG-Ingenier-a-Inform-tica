# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:16:15 2026

@author: TESTER
"""
import itertools
from pathlib import Path
import pandas as pd
def get_all_csvs(path : Path, **kwargs):
    return path.glob("*.csv")#.iterdir()

chartsPath=Path("Resultados/EnsembleEDDM")

resultsPath=Path("Resultados/EnsembleEDDM/Rendimiento")
freqsPath=Path("Resultados/EnsembleEDDM/Frecuencias")
dictsPath=Path("Resultados/EnsembleEDDM/Diccionarios")

resultsFileName="EnsembleEDDMPerformance"
mergedResultsPath=chartsPath.joinpath(resultsFileName+".csv")
freqsFileName="EnsembleEDDMFeatureFrequencies"
mergedFreqsPath=chartsPath.joinpath(freqsFileName+".csv")
dictsFileName="EnsembleEDDMDictionaries"
mergedDictsPath=chartsPath.joinpath(dictsFileName+".csv")
def merge_all_csvs(path : Path, csvBuffer : list, **kwargs):
    fileIterator=get_all_csvs(path)
    csvBuffer.extend(fileIterator)
    fileIterator=get_all_csvs(path)
    #numberOfFiles=len(csvBuffer)
    #print(numberOfFiles)
    i=0
    for csvFile in fileIterator:
        csvAsDf=pd.read_csv(csvFile, **kwargs)
        csvBuffer[i]=csvAsDf
        i+=1
        #print(i)    
        
#names=itertools.product([0,1,2],repeat,6)
frames=[]
dicts=[]
freqs=[]
merge_all_csvs(resultsPath, frames)
merge_all_csvs(freqsPath, freqs, header=0, index_col=[0,1,2,3,4,5,6,7])
merge_all_csvs(dictsPath, dicts, header=[0,1,2,3,4,5], index_col=[0,1,2,3,4,5,6,7])

results=pd.concat(frames)
frequencies=pd.concat(freqs)
dictionaries=pd.concat(dicts)
freqsFinal=pd.DataFrame(data=frequencies.values[:,::2], columns=list(itertools.product([0,1,2],repeat=6)),index=dictionaries.index)
dictsFinal=pd.DataFrame(data=dictionaries.values, columns=list(itertools.product([0,1,2],repeat=6)), index=dictionaries.index)
print(results, frequencies, dictionaries)
results.to_csv(mergedResultsPath)
freqsFinal.to_csv(mergedFreqsPath)
dictsFinal.to_csv(mergedDictsPath)