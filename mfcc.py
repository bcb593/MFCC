import librosa as lr
from glob import glob
import pylab
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
data='/Users/bidhanbashyal/Acoustic/Airport Data1/audio'
audio_file=glob(data+'/*.wav')
#Jusst to verify where the data is loading or not. 
len(audio_file)

for file in range(0, len(audio_file), 1):
    x, sr = lr.load(audio_file[file])
    mfccs = librosa.feature.mfcc(x, sr=sr)
   # print(mfccs.shape)
    save_path = '/Users/bidhanbashyal/Acoustic/Airport Data1/MFCC/'
    filename = str(file)+".jpg"
    savepath = os.path.join(save_path, filename)
    fig=librosa.display.specshow(mfccs, sr=440000, x_axis='time')
    pylab.savefig(save_path+filename, bbox_inches=None, pad_inches=0)
    
    
    
    
    
    
    #Just to play the audio sound
    import IPython.display as ipd
    ipd.Audio(audio_file[600])
