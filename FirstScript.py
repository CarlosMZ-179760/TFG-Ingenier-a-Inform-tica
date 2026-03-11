# -*- coding: utf-8 -*-
"""
Un primer script de prueba breve para el detector ensemble 
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

import capymoa.drift.detectors.ensemble_detector as ensemble
print(sys.path)
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
print(detector.get_all_base_detections())
#print(detector.get_all_base_warnings())