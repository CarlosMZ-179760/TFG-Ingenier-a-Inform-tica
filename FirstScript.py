# -*- coding: utf-8 -*-
"""
Un primer script de prueba breve para el detector ensemble 
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from capymoa.stream import NumpyStream
from capymoa.instance import LabeledInstance
from capymoa.classifier import OnlineBagging
import capymoa.stream.generator as streamGen

#from moa.streams.generators.cd import GenericChangeGenerator as MOA_GenericChangeGenerator
import capymoa.drift.detectors.ensemble_detector as ensemble
import capymoa.stream.generator as streamGen
import os, pprint, sys

valid_delay=500
durationChange=0
durationStableConcept=5000
apparentDelay=durationChange+valid_delay
driftsList=[20]
print(driftsList, flush=True)

generator=streamGen.GenericChangeGenerator(valid_delay=apparentDelay, number_of_drifts=10, duration_change=durationChange, instance_random_seed=3, duration_stable_concept=durationStableConcept)
# i=generator.next_instance()
# #print("First instance:", i)
# #print(type(generator.next_instance()))
# t=np.array([])
# d=np.array([])
# for i in range (59000):
#     y=generator.next_instance().x
#     #print(y)
#     #if (y[1]==1):
#     #    print(y[1])
    
#     #print(y)
#     t=np.append(t, y[0])
#     d=np.append(d,y[1])
#     #print("j=",i)
# print(sum(d))
# plt.plot(d)
# window_size = 1000

# i = 0
# # Initialize an empty list to store moving averages
# moving_averages = np.zeros(t.shape)

# # Loop through the array t o
# #consider every window of size 3
# while i < len(t) - window_size + 1:
#     #print("i=",i)
#     # Calculate the average of current window
#     window_average = np.sum(t[
#       i:i+window_size]) / window_size
    
#     # Store the average of current
#     # window in moving average list
#     moving_averages[i+window_size-1]=window_average
    
#     # Shift window to right by one position
#     i += 1

# plt.plot(moving_averages)
# print(generator)
# data_stream_generator = np.random.default_rng(seed=179760)
# data_stream=np.absolute(data_stream_generator.normal(loc=0, scale=1, size=2000))
# for i in range(999, 2000):
#     data_stream[i] = np.absolute(data_stream_generator.normal(loc=0, scale=2))
# data_stream_average=[]
# rang = 10
# x=np.arange(rang,2000)
# for ind in range(rang):
#     data_stream_average.append(np.mean(data_stream[:ind]))
# for ind in range(len(data_stream)-rang +1):
#     data_stream_average.append(np.mean(data_stream[ind:ind+rang]))
# #plt.plot(x,data_stream_average[rang+1:])

#trues=np.zeros(2*(drifts+1)*(durationStableConcept+durationChange)-durationChange-delay)
for drifts in driftsList:
    i=0
    sys.stdout = open('output-'+str(drifts)+'-Oza(NB).txt','w')
    print("\n\n\n\n NUEVA EJECUCIÓN \n \n\n")
    start=datetime.datetime.now()
    print(start)
    print("Delay, DOC, DSC, Drifts=", apparentDelay,durationChange,durationStableConcept, drifts)
    readings=np.zeros([2*(drifts+1)*(durationStableConcept+durationChange)-durationChange+apparentDelay,1])
    driftHistory=np.zeros([2*(drifts+1)*(durationStableConcept+durationChange)-durationChange+apparentDelay])
    #print("\n")
    datasetGenerator=streamGen.GenericChangeGenerator(valid_delay=apparentDelay, number_of_drifts=drifts, duration_change=durationChange, instance_random_seed=30, duration_stable_concept=durationStableConcept)
    while datasetGenerator.has_more_instances():
        stream_elem=datasetGenerator.next_instance().x
        atom=stream_elem[0]
        y=stream_elem[1]
        #print("y=", y)
        readings[i]=atom
        driftHistory[i]=y
        i+=1
    #print(readings, drifts)
    trainingStream : NumpyStream[LabeledInstance]=NumpyStream(readings, driftHistory, dataset_name='trainingStream', feature_names=['classifierError'], target_name='Drift', target_type='categorical')
    detector = ensemble.EnsembleDetector(dataset=trainingStream, valid_delay=apparentDelay, datasetSize=np.size(driftHistory))
    detector.pretrain()
    detector.describe_drifts()
    testGenerator=streamGen.GenericChangeGenerator(valid_delay=apparentDelay, number_of_drifts=drifts, duration_change=durationChange, instance_random_seed=10, duration_stable_concept=durationStableConcept)
    trueDriftsTest=[]
    i=1
    for instance in testGenerator:
        detector.add_element(instance.x[0])
        if instance.x[1]==1:
            trueDriftsTest.append(i)
        i+=1
    #detector.add_element(1)
    print(trueDriftsTest, detector.get_all_detections())
    end=datetime.datetime.now()
    print(end)
    print(end-start,(end-start)/drifts)
#print(detector.training_report)
#for i in range(2000):
#    detector.add_element(data_stream[i])
#    print(detector.get_states())
#print(detector.get_all_base_detections())
#print(detector.get_all_base_warnings())