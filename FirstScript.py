# -*- coding: utf-8 -*-
"""
Un primer script de prueba breve para el detector ensemble 
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

#from moa.streams.generators.cd import GenericChangeGenerator as MOA_GenericChangeGenerator
import capymoa.drift.detectors.ensemble_detector as ensemble
import capymoa.stream.generator as streamGen
import os, pprint, sys

generator=streamGen.GenericChangeGenerator(duration_change=1000, instance_random_seed=3, duration_stable_concept=5000)
i=generator.next_instance()
#print("First instance:", i)
#print(type(generator.next_instance()))
t=np.array([])
d=np.array([])
for i in range (59000):
    y=generator.next_instance().x
    #print(y)
    #if (y[1]==1):
    #    print(y[1])
    
    #print(y)
    t=np.append(t, y[0])
    d=np.append(d,y[1])
    #print("j=",i)
print(sum(d))
plt.plot(d)
window_size = 1000

i = 0
# Initialize an empty list to store moving averages
moving_averages = np.zeros(t.shape)

# Loop through the array t o
#consider every window of size 3
while i < len(t) - window_size + 1:
    #print("i=",i)
    # Calculate the average of current window
    window_average = np.sum(t[
      i:i+window_size]) / window_size
    
    # Store the average of current
    # window in moving average list
    moving_averages[i+window_size-1]=window_average
    
    # Shift window to right by one position
    i += 1

plt.plot(moving_averages)
print(generator)
data_stream_generator = np.random.default_rng(seed=179760)
data_stream=np.absolute(data_stream_generator.normal(loc=0, scale=1, size=2000))
for i in range(999, 2000):
    data_stream[i] = np.absolute(data_stream_generator.normal(loc=0, scale=2))
data_stream_average=[]
rang = 10
x=np.arange(rang,2000)
for ind in range(rang):
    data_stream_average.append(np.mean(data_stream[:ind]))
for ind in range(len(data_stream)-rang +1):
    data_stream_average.append(np.mean(data_stream[ind:ind+rang]))
#plt.plot(x,data_stream_average[rang+1:])


detector = ensemble.EnsembleDetector()
for i in range(2000):
    detector.add_element(data_stream[i])
    #print(detector.get_states())
#print(detector.get_all_base_detections())
#print(detector.get_all_base_warnings())