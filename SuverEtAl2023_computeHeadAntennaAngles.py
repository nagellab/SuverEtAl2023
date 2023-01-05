"""
Compute antenna angles (including head axis) from tracked video data
For Suver, Medina & Nagel 2023

"""

####################################################
# Dependencies
####################################################

from pathlib import Path
import argparse
from deeplabcut.utils import auxiliaryfunctions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import pandas as pd
import os
import cv2
#from scipy import ndimage
from skimage.draw import circle_perimeter, circle
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import ImportMat_DataStruct as im
import SuverEtAl2023_constants as const


#pull out X,Y-positions of 'bodyparts' (antenna features) of interest
def getXY_trackedBodyparts(config_path, videofolder, vidName):
    cfg = auxiliaryfunctions.read_config(config_path)
    bodyparts = cfg['bodyparts']

    #read in tracking data frame
    trainFraction = cfg['TrainingFraction'][0]
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,1,trainFraction) #automatically loads corresponding model (even training iteration based on snapshot index)
    dataname = os.path.join(str(videofolder),vidName+DLCscorer + '.h5')
    if os.path.isfile(dataname):
        Dataframe = pd.read_hdf(dataname)
    else:
        return [], []
    nframes = len(Dataframe.index)
    df_likelihood = np.empty((len(bodyparts),nframes))
    df_x = np.empty((len(bodyparts),nframes))
    df_y = np.empty((len(bodyparts),nframes))
    for bpind, bp in enumerate(bodyparts):
        df_likelihood[bpind,:]=Dataframe[DLCscorer][bp]['likelihood'].values
        df_x[bpind,:]=Dataframe[DLCscorer][bp]['x'].values
        df_y[bpind,:]=Dataframe[DLCscorer][bp]['y'].values

    return df_x, df_y

# returns x, y postition of anterior and posterior head axis
# computed using the first video in an experiment, and estimated using all frames (averaged)
# HARD-CODED folder path (may want to make this a global variable in the future!)
def get_head_axis(expt = '2020_12_14_E3', cameraView = "frontal", frameNumToPlot = 1, TEST = 0):

    trialNum = 1#2 #base head axis on first video
     #frames to plot head axis on (single or average across)
    videotype = 'avi'

    if (cameraView == 'frontal'):
        vidName = expt+'_Video_frontal_'+str(trialNum)
        config_path = const.config_path['frontal']
        videofolder = const.baseDirectory+'/SuverEtAl2023_Data/videos_frontal'

    video_path = videofolder+'/'+vidName+'.'+videotype
    cfg = auxiliaryfunctions.read_config(config_path)
    bodyparts = cfg['bodyparts']

    df_x, df_y = getXY_trackedBodyparts(config_path, videofolder, vidName)

    # compute and plot head axis coordinates
    markers = const.headAxisMarkers[cameraView]
    xant = np.mean([np.mean(df_x[markers[0][0]]),np.mean(df_x[markers[1][0]])])
    yant = np.mean([np.mean(df_y[markers[0][0]]),np.mean(df_y[markers[1][0]])])
    xpost = np.mean([np.mean(df_x[markers[0][1]]),np.mean(df_x[markers[1][1]])])
    ypost = np.mean([np.mean(df_y[markers[0][1]]),np.mean(df_y[markers[1][1]])])
    #option to plot tracked points and head axis for one frame
    if TEST:
        colormap = const.colormap
        colorclass=plt.cm.ScalarMappable(cmap=colormap)
        C=colorclass.to_rgba(np.linspace(0,1,len(bodyparts)))
        colors=(C[:,:3]*255).astype(np.uint8)
        capVid = cv2.VideoCapture(video_path)
        print(video_path)
        capVid.set(1,frameNumToPlot)
        ret, frame = capVid.read() #read the frame
        ny = capVid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        nx= capVid.get(cv2.CAP_PROP_FRAME_WIDTH)

        for bpind, bp in enumerate(bodyparts):
            xc = int(df_x[bpind,frameNumToPlot])
            yc = int(df_y[bpind,frameNumToPlot])
            rr, cc = circle(yc,xc,const.dotsize_bodypart,shape=(ny,nx))
            frame[rr, cc, :] = colors[bpind]

        fig, ax = plt.subplots(facecolor=(0.1,0.1,0.1),figsize=(20,18))
        #plot head axis points
        ax.plot(xant, yant,marker='o',color='white',markersize='2')
        #ax.plot(xant, yant+50,marker='o',color='pink',markersize='1')
        ax.plot(xpost, ypost,marker='o',color='white',markersize='2')
        #draw and lengthen head axis for easier visualization!
        rise = (yant-ypost)
        run = (xant-xpost)
        if (cameraView == 'frontal'):
            extAnt = 3
            extPost = 5
        ax.plot([xant+run*extAnt,xpost-run*extPost],[yant+rise*extAnt,ypost-rise*extPost],linewidth=1,color='white', linestyle='--')
        plt.title('Defining head axis!',color='white')
        ax.imshow(frame)
        plt.pause(0.001)
        plt.show(block=False)
        plt.show()
    else:
        ax = []
    return [xant, yant, xpost, ypost, ax, plt]

#returns an m x n list where m = #frames and n = # antenna angles
def compute_angles(xant,yant,xpost,ypost,df_x,df_y,frameNumsToComputeAngle,angpairs,ax,cfg, frameNumToPlot, cameraView):
    TEST = 1
    wid = len(angpairs)
    length = len(frameNumsToComputeAngle)
    angles = np.zeros((length,wid))
q
    # iterate over each pair of interest and compute its angle relative to the midline of the head
    for ii in range(len(angpairs)):
        pairind = angpairs[ii]
        bpind1 = pairind[0]
        bpind2 = pairind[1]
        for frameInd in range(len(frameNumsToComputeAngle)):
            frameNum = frameNumsToComputeAngle[frameInd]
            x1 = int(df_x[bpind1,frameNum])
            y1 = int(df_y[bpind1,frameNum])
            bodyparts = cfg['bodyparts']
            part = bodyparts[bpind1] #grab name of bodypart (determine if left/right)
            x2 = int(df_x[bpind2,frameNum])
            y2 = int(df_y[bpind2,frameNum])
            #compute angle of part relative to midline - positive values move away from midline
            x4, y4 = xant, yant
            x3, y3 = xpost, ypost

            #compute relative to anterior (e.g. downward antennal deflection produces l and r positive values)
            if part[0] == 'l':
                ang1 = np.arctan2((y2-y1), (x2-x1))
                angHeadAxis = np.arctan2((y4-y3), (x4-x3))
                angRad = np.abs(angHeadAxis)-np.abs(ang1)
            else:
                ang1 = np.arctan2((y1-y2), (x1-x2))
                angHeadAxis = np.arctan2((y3-y4), (x3-x4))
                if (cameraView == "frontal"):
                    angRad = -(np.abs(ang1)-np.abs(angHeadAxis))
                else:
                    angRad = np.abs(ang1)-np.abs(angHeadAxis)

            angDeg = round(angRad*(180/np.pi),1)
            angles[frameInd][ii] = angDeg
    return angles

#returns an m x n list where m = #frames and n = # antenna angles
def get_antenna_angles(expt = '2020_12_14_E3', cameraView = 'frontal', trialNum = 1, TEST = 0):
    frameNumToPlot = 4#400
    #grab head axis and plot
    [xant, yant, xpost, ypost, ax, plt] = get_head_axis(expt, cameraView, frameNumToPlot, TEST)

    videotype = 'avi'
    if (cameraView == 'frontal'):
        vidName = expt+'_Video_frontal_'+str(trialNum)
        videofolder = const.baseDirectory+'SuverEtAl2023_Data/videos_frontal'

    config_path = const.config_path[cameraView]
    cfg = auxiliaryfunctions.read_config(config_path)
    bodyparts = cfg['bodyparts']

    [df_x, df_y] = getXY_trackedBodyparts(config_path, videofolder, vidName)
    if df_x == []:
        return []

    nframes = len(df_x[0])
    frameNumsToComputeAngle = list(range(0,nframes))
    angPairs = const.angPairs[cameraView]
    angles = compute_angles(xant,yant,xpost,ypost,df_x,df_y,frameNumsToComputeAngle,angPairs,ax,cfg, frameNumToPlot, cameraView)
    #plot line across pairs of points defining angles
    if TEST:
        plt.title('Antenna angles',color='white')
        colormap = const.colormap
        dotsize = 3.5
        colorclass=plt.cm.ScalarMappable(cmap=colormap)
        C=colorclass.to_rgba(np.linspace(0,1,len(bodyparts)))
        colors=(C[:,:3]*255).astype(np.uint8)
        for ap in angPairs:
            bp1 = ap[0]
            bp2 = ap[1]
            ax.plot([df_x[bp1,frameNumToPlot],df_x[bp2,frameNumToPlot]],[df_y[bp1,frameNumToPlot],df_y[bp2,frameNumToPlot]],linewidth=1,color='white', linestyle='--')

        fig, ax = plt.subplots(1,1,facecolor=const.figColor,figsize=(12,8))
        print(np.shape(angles))
        plt.plot(angles[:,0])
        plt.plot(angles[:,3])
        plt.plot(angles[:,4])
        plt.plot(angles[:,7])
        plt.show()
        plt.pause(0.001)
        plt.show(block=False)

    return angles
