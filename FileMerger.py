# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:16:15 2026

@author: TESTER
"""
from pathlib import Path
import pandas as pd
def get_all_csvs(path : Path, **kwargs):
    return path.glob("*.csv")#.iterdir()

chartsPath=Path("Resultados/Ensemble")

resultsPath=Path("Resultados/Ensemble/Rendimiento")
freqsPath=Path("Resultados/Ensemble/Frecuencias")
dictsPath=Path("Resultados/Ensemble/Diccionarios")

resultsFileName="EnsemblePerformance"
mergedResultsPath=chartsPath.joinpath(resultsFileName+".csv")
freqsFileName="EnsembleFeatureFrequencies"
mergedFreqsPath=chartsPath.joinpath(freqsFileName+".csv")
dictsFileName="EnsembleDictionaries"
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
        
frames=[]
dicts=[]
freqs=[]
merge_all_csvs(resultsPath, frames)
merge_all_csvs(freqsPath, freqs, header=[0,1,2,3,4], index_col=[0,1,2,3,4,5,6,7])
merge_all_csvs(dictsPath, dicts, header=[0,1,2,3,4], index_col=[0,1,2,3,4,5,6,7])

results=pd.concat(frames)
frequencies=pd.concat(freqs)
dictionaries=pd.concat(dicts)
print(results, frequencies, dictionaries)
results.to_csv(mergedResultsPath)
frequencies.to_csv(mergedFreqsPath)
dictionaries.to_csv(mergedDictsPath)