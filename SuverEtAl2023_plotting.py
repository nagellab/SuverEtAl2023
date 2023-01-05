"""
Plotting scripts for Suver, Medina & Nagel 2023

"""

####################################################
# Dependencies
####################################################

from pathlib import Path
import argparse
from deeplabcut.utils import auxiliaryfunctions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date
from scipy import signal
import statsmodels.api as sm
from numpy.random import rand
from math import log10, floor
from scipy import stats
import cv2
from skimage.draw import circle_perimeter, circle
from scipy.io import savemat

import ImportMat_DataStruct as im
import SuverEtAl2023_computeHeadAntennaAngles as ch
import SuverEtAl2023_constants as const



def set_axis_standard_preferences(ax):
    # configure the plot!
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    return ax

def getSingleTrackingSet(expt = '2018_09_06_E4',trialNum = 29, cameraView='frontal'):

    cfg_path = const.config_path[cameraView] # '../frontal_velocityTuning-Marie-2019-09-19/config.yaml'

    videofolder = const.baseDirectory+'../SuverEtAl_Data/videos_'+cameraView
    cfg = auxiliaryfunctions.read_config(cfg_path)
    trackedPoints=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,'all')

    trainFraction = cfg['TrainingFraction'][0]
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,1,trainFraction)  # automatically loads corresponding model (even training iteration based on snapshot index)
    if (cameraView == 'frontal'):
        dataname = os.path.join(str(videofolder),expt+'_Video_frontal_'+str(trialNum)+DLCscorer + '.h5')
    Dataframe = pd.read_hdf(dataname)
    nframes = len(Dataframe.index)
    df_likelihood = np.empty((len(trackedPoints),nframes))
    df_x = np.empty((len(trackedPoints),nframes))
    df_y = np.empty((len(trackedPoints),nframes))
    for bpind, bp in enumerate(trackedPoints):
        df_likelihood[bpind,:]=Dataframe[DLCscorer][bp]['likelihood'].values
        df_x[bpind,:]=Dataframe[DLCscorer][bp]['x'].values
        df_y[bpind,:]=Dataframe[DLCscorer][bp]['y'].values

    return df_x, df_y

#this will plot all of the tracked points (not angles)
def plotSingleRawBodypart(expt = '2020_12_14_E3', trialNum = 1, cameraView='frontal'):
    cfg_path = const.config_path[cameraView]# '../frontal_velocityTuning-Marie-2019-09-19/config.yaml'
    fig, ax = plt.subplots(facecolor=(1,1,1),figsize=(8,6))
    cfg = auxiliaryfunctions.read_config(cfg_path)
    trackedPoints=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,'all')
    df_x, df_y = getSingleTrackingSet(expt, trialNum, cameraView)
    colormap = 'jet'
    colorclass=plt.cm.ScalarMappable(cmap=colormap)
    C=colorclass.to_rgba(np.linspace(0,1,len(trackedPoints)))
    colors=(C[:,:3]*255).astype(np.uint8)

    for bpind, bp in enumerate(trackedPoints):
        ax.plot(df_x[bpind,:],color=colors[bpind]/255)
        ax.plot(df_y[bpind,:],color=colors[bpind]/255, linestyle='--')

    #videofolder = const.baseDirectory+'Data_'+cameraView
    matfolder = const.matDirectory
    matfilename = matfolder+'/'+expt+'.mat'
    exptNotes = im.importMat_mn_velTuning(matfilename)

    plt.title('Single fly raw body part position\n'+expt+' trial '+str(trialNum)+', vel= '+str(exptNotes.velocity[trialNum-1])+', valve'+str(exptNotes.valveState[trialNum-1]))
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    return

# extract cell body side for this experiment (side of whole cell patch recording)
# uses manually-generated set of notes for all experiments used in this analysis
def get_expt_notes(expt, cellBodySide=0):

    import loadNotes_mnOptoVelocityTuning as ln
    for noteInd, notes in enumerate(ln.all_notes):
        exptNotes = list(filter(lambda ex: ex['date'] == expt, notes))
        # if list is not empty, we have found our experiment!
        if exptNotes:
            exptNotes = exptNotes[0]
            if cellBodySide == 1:
                cb_side = exptNotes['cb_side']
                return cb_side
            else:
                return exptNotes

    print('Cannot find '+expt+'in load_notes_activePassive notes; check experiment name!')
    return 'error finding expt name!'

# For specified experiment ('expt') return class of experiment it is from (e.g. 'CS_odor')
def get_expt_type(expt):
    import loadNotes_mnOptoVelocityTuning as ln
    for noteInd, notes in enumerate(ln.all_notes):
        exptNotes = list(filter(lambda ex: ex['date'] == expt, notes))
        # if list is not empty, we have found our experiment!
        if exptNotes:
            exptNotes = exptNotes[0]
            return exptNotes['cell_type']

def test_second_segment(expt='2021_03_10_E5',cameraView='frontal',trialNum=43):
    angles = get_antenna_angles_adjusted(expt, cameraView, trialNum)
    print('Right 2nd:')
    print(angles[:,0])
    print('Right 3rd:')
    print(angles[:,1])
    fig, ax = plt.subplots(1,1,facecolor=const.figColor,figsize=(12,8))
    plt.plot(angles[:,0])
    plt.plot(angles[:,1])
    plt.plot(angles[:,2])
    plt.plot(angles[:,3])
    plt.show()
    plt.pause(0.001)
    plt.show(block=False)

# average across defined sets of angles for increased signal-to-noise
# This will usually be done for the second antennal segments
# Groups of points to average across are defined in constants_activeMovements
def get_antenna_angles_adjusted(expt, cameraView, trialNum):
    angles = ch.get_antenna_angles(expt, cameraView, trialNum, 0)
    if angles == []:
        print('Tracking for this video (or this video, '+expt+cameraView+'trialNum '+str(trialNum)+') does not exist, returning.')
        return []
    #custom averaging for more accurate estimates of antennal movements
    #WARNING: HARD-CODED!
    newAngles = np.zeros((np.shape(angles)[0],4)) #hard-coded [4]...

    if (cameraView == 'frontal'):
        cam = 'frontal'
        rightAvgSet = const.angPairAverageSets[cam][0]
        leftAvgSet = const.angPairAverageSets[cam][1]
        secondSegmentAverage_right = np.nanmean(angles[:,rightAvgSet],1)
        secondSegmentAverage_left = np.nanmean(angles[:,leftAvgSet],1)
        newAngles[:,0] = secondSegmentAverage_right
        if(cameraView == 'frontal'):
            newAngles[:,0] = secondSegmentAverage_right
            newAngles[:,1] = angles[:,2]
            newAngles[:,2] = secondSegmentAverage_left
            newAngles[:,3] = angles[:,5]
        #grab notes associated with this experiment
        videofolder = const.baseDirectory+'Data_frontal'
    matfolder = const.matDirectory
    matfilename = matfolder+'/'+expt+'.mat'
    exptNotes = im.importMat_velTuning(matfilename)

    framerate = exptNotes.fps[0]
    numPreInds = int(exptNotes.pre_trial_time[0]*framerate)
    baselineInds = list(range(0,numPreInds))
    if (const.omitDarkFrames == 1) & (cameraView != 'frontal'): #& (cameraView != 'frontal'):
        baselineInds = baselineInds[const.trimStart:]
        newAngles[0:const.trimStart] = np.nan

    return newAngles

# lines = -1 gives you plain image with no tracking
# lines = 0 gives just tracking bodyparts (points imbedded in image)
# lines = 1 gives points used to determine angles (points imbedded in image) + angles
# lines = 2 gives just antenna angles (lines over image)
# lines = 3 plots angles from points used for angles (e.g. multiple 2nd segment hairs)
# headaxis = 0 shows no head axis, headaxis = 1 shows axis determined from cephalic hairs
def plot_head_markers_axis(expt='2020_12_14_E3', trialNum=1, frameNum=1, lines=0, headaxis = 0,savefig=1):
    lineWid = 6
    cameraView='frontal' #hard-coded for now
    exptNotes = get_mat_notes(expt)
    windVel = exptNotes.velocity[trialNum-1]
    #print(exptNotes.velocity[:],exptNotes.valveState[:])
    windVel = windVel[0] #take values out of brackets
    valveStates = exptNotes.valveState[trialNum-1]
    windDir = valveStates[0]/2-1 #convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)

    import computeHeadAntennaAngles as ch
    # grab head axis markers
    [xant, yant, xpost, ypost, ax, plt] = ch.get_head_axis(expt, cameraView, trialNum, TEST=0);

    videofolder = const.baseDirectory+'SuverEtAl_Data/videos_'+cameraView
    videotype = 'avi'
    if cameraView == 'frontal':
        vidName = expt+'_Video_frontal_'+str(trialNum)

    config_path = const.config_path[cameraView]
    video_path = videofolder+'/'+vidName+'.'+videotype
    print(video_path)
    cfg = auxiliaryfunctions.read_config(config_path)
    bodyparts = cfg['bodyparts']

    [df_x, df_y] = ch.getXY_trackedBodyparts(config_path, videofolder, vidName)

    nframes = len(df_x[0])
    frameNumsToComputeAngle = list(range(0,nframes))
    angPairs = const.angPairs[cameraView]
    angles = get_antenna_angles_adjusted(expt, cameraView, trialNum)

    figAng, axAng = plt.subplots(1,1,facecolor=const.figColor,figsize=(12,8))
    plt.plot(angles[:,0])
    plt.plot(angles[:,1])
    plt.plot(angles[:,2])
    plt.plot(angles[:,3])
    plt.show()
    plt.pause(0.001)
    plt.show(block=False)

    colormap = const.colormap
    colorclass=plt.cm.ScalarMappable(cmap=colormap)
    C=colorclass.to_rgba(np.linspace(0,1,len(bodyparts)))
    colors=(C[:,:3]*255).astype(np.uint8)
    capVid = cv2.VideoCapture(video_path)

    capVid.set(1,frameNum-1)
    ret, frame = capVid.read() #read the frame
    ny = capVid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    nx= capVid.get(cv2.CAP_PROP_FRAME_WIDTH)

    fig, ax = plt.subplots(facecolor=(0.1,0.1,0.1),figsize=(20,18))

    print(const.bodypartsOfInterest[cameraView])
    if lines == 0:
        for bpind, bp in enumerate(bodyparts):
            xc = int(df_x[bpind,np.int(frameNum-1)])
            yc = int(df_y[bpind,np.int(frameNum-1)])
            rr, cc = circle(yc,xc,const.dotsize_bodypart,shape=(ny,nx))
            frame[rr, cc, :] = colors[bpind]
    elif lines == -2:
        for bpind in const.bodypartsOfInterestWithFun[cameraView]:
            print(bpind)
            xc = int(df_x[bpind,np.int(frameNum-1)])
            yc = int(df_y[bpind,np.int(frameNum-1)])
            rr, cc = circle(yc,xc,const.dotsize_bodypart,shape=(ny,nx))
            frame[rr, cc, :] = colors[bpind]
    elif lines == 1:
        for bpind in const.bodypartsOfInterest[cameraView]:
            print(bpind)
            xc = int(df_x[bpind,np.int(frameNum-1)])
            yc = int(df_y[bpind,np.int(frameNum-1)])
            rr, cc = circle(yc,xc,const.dotsize_bodypart,shape=(ny,nx))
            frame[rr, cc, :] = colors[bpind]

    #draw and lengthen head axis for easier visualization!
    if headaxis == 1:
        #plot head axis points
        ax.plot(xant, yant,marker='o',color='white',markersize='2')
        ax.plot(xpost, ypost,marker='o',color='white',markersize='2')
        rise = (yant-ypost)
        run = (xant-xpost)
        if (cameraView == 'frontal'):
            extAnt = 3
            extPost = 5

        ax.plot([xant+run*extAnt,xpost-run*extPost],[yant+rise*extAnt,ypost-rise*extPost],linewidth=lineWid,color='white', linestyle='--')

    rightAvgSet = const.angPairAverageSets[cameraView][0]
    leftAvgSet = const.angPairAverageSets[cameraView][1]
    # x positions of right antenna angles
    rightAngles_x = np.zeros((4)) #hard-coded [4]...
    rightAngles_x[0] = np.nanmean(df_x[[5,7], frameNum-1])
    rightAngles_x[1] = np.nanmean(df_x[[6,8], frameNum-1])
    rightAngles_x[2] = np.nanmean(df_x[12, frameNum-1])
    rightAngles_x[3] = df_x[13, frameNum-1]
    # y positions of right antenna angles
    rightAngles_y = np.zeros((4)) #hard-coded [4]...
    rightAngles_y[0] = np.nanmean(df_y[[5,7], frameNum-1])
    rightAngles_y[1] = np.nanmean(df_y[[6,8], frameNum-1])
    rightAngles_y[2] = np.nanmean(df_y[12, frameNum-1])
    rightAngles_y[3] = df_y[13, frameNum-1]

    # x positions of right antenna angles
    leftAngles_x = np.zeros((4)) #hard-coded [4]...
    leftAngles_x[0] = np.nanmean(df_x[[21,23], frameNum-1])
    leftAngles_x[1] = np.nanmean(df_x[[22,24], frameNum-1])
    leftAngles_x[2] = np.nanmean(df_x[28, frameNum-1])
    leftAngles_x[3] = df_x[29, frameNum-1]
    # y positions of right antenna angles
    leftAngles_y = np.zeros((4)) #hard-coded [4]...
    leftAngles_y[0] = np.nanmean(df_y[[21,23], frameNum-1])
    leftAngles_y[1] = np.nanmean(df_y[[22,24], frameNum-1])
    leftAngles_y[2] = np.nanmean(df_y[28, frameNum-1])
    leftAngles_y[3] = df_y[29, frameNum-1]

    if lines == 2:
        ax.plot([rightAngles_x[0],rightAngles_x[0+1]],[rightAngles_y[0],rightAngles_y[0+1]],linewidth=lineWid,color=const.colors_antAngs[0], linestyle='-')
        ax.plot([leftAngles_x[0],leftAngles_x[0+1]],[leftAngles_y[0],leftAngles_y[0+1]],linewidth=lineWid,color=const.colors_antAngs[2], linestyle='-')
        ax.plot([rightAngles_x[2],rightAngles_x[2+1]],[rightAngles_y[2],rightAngles_y[2+1]],linewidth=lineWid,color=const.colors_antAngs[1], linestyle='-')
        ax.plot([leftAngles_x[2],leftAngles_x[2+1]],[leftAngles_y[2],leftAngles_y[2+1]],linewidth=lineWid,color=const.colors_antAngs[3], linestyle='-')

    elif (lines == 1) | (lines == 3): #will plot angles of
        for ap in angPairs:
            bp1 = ap[0]
            bp2 = ap[1]
            ax.plot([df_x[bp1,frameNum-1],df_x[bp2,frameNum-1]],[df_y[bp1,frameNum-1],df_y[bp2,frameNum-1]],linewidth=lineWid,color='pink', linestyle='--')

    plt.title(expt+' trial '+str(trialNum)+ ' frame '+str(frameNum)+' vel= '+str(windVel)+ ' dir= '+ str(windDir),color='white',loc='left')

    ax.imshow(frame)
    plt.show()
    plt.pause(0.001)
    plt.show(block=False)

    if savefig==1:
        fig = plt.figure(1)#plt.gcf()
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath + 'videoStillWithHeadMarkers_' + expt+ '_trial_'+str(trialNum)+ ' frame '+str(frameNum)+' vel= '+str(windVel)+ ' dir= '+ str(windDir)+ '_'+str(lines)+'.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + 'videoStillWithHeadMarkers_' + expt+ '_trial_'+str(trialNum)+ ' frame '+str(frameNum)+' vel= '+str(windVel)+ ' dir= '+ str(windDir)+ '_'+str(lines)+'.png'
        fig.savefig(savepath, facecolor=fig.get_facecolor())


def get_mat_notes(expt):
    matfolder = const.matDirectory
    matfilename = matfolder+expt+'.mat'
    exptNotes = im.importMat_velTuning(matfilename)
    return exptNotes

# gather (and save if importing anew) information related to flight:
# returns whether the fly is flying or not (isFlying), and when (overThresh)
def getFlightsSingleExpt(expt = '2021_09_09_E1', cameraView = 'frontal', importAnew = 0):

    exptNotes = get_mat_notes(expt)
    numTrials = np.shape(exptNotes)[0]
    stimStart = exptNotes.pre_trial_time[0][0]
    stimStop = exptNotes.pre_trial_time[0][0]+exptNotes.trial_time_wind[0][0]
    windStart =stimStart*const.samplerate
    windStop = stimStop*const.samplerate

    flights = np.empty([numTrials], dtype=bool)
    isFlying = np.empty([numTrials], dtype=float)
    flyingPercent = np.empty([numTrials], dtype=float)
    tachOverThreshPercent = np.empty([numTrials])
    overThresh = np.empty([numTrials,const.framerate*exptNotes.numSecOut[0][0]])
    dsamp = int(np.ceil(const.samplerate/const.framerate))
    # angs_all will contain all traces relative to midline for this experiment
    angs_all = np.empty([numTrials, const.framerate*const.lenVideo,
        np.shape(const.angPairNames[cameraView])[0]])
    bb,aa = generate_butter_filter(samplerate = const.samplerate) #make filter for analyzing tachometer signal
    saved_overThresh_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'overThresh.npy'
    for idx in range(1,numTrials+1):
        saved_angle_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'angles.npy'
        if os.path.isfile(saved_angle_fn):
            angles = np.load(saved_angle_fn) #generally do not import this anew here - focus on flight data
        else:
            angles = get_antenna_angles_adjusted(expt, cameraView, idx)
            np.save(saved_angle_fn, angles)
        saved_flight_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'flights.npy'
        saved_isFlying_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'isFlying.npy'
        saved_flyingPercent_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'flyingPercent.npy'
        saved_tach_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'tach.npy'
        saved_overThreshFly_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'overThreshFly.npy'
        if ~importAnew & os.path.isfile(saved_flight_fn): #check if flight data been loaded and saved previously
            flights = np.load(saved_flight_fn)
            isFlying = np.load(saved_isFlying_fn)
            flyingPercent = np.load(saved_flyingPercent_fn)
            tachOverThreshPercent = np.load(saved_tach_fn)
            overThresh = np.load(saved_overThresh_fn)
        else: #load notes from scratch (first time analyzing!)
            tach = exptNotes.tachometer[idx-1]
            tachOverThresh, overThreshFly = detect_flight(tach, bb, aa)
            overThresh[idx-1] = overThreshFly[0:-1:dsamp]
            isInWindRegion = 1*np.array([(tachOverThresh>=windStart) & (tachOverThresh<=windStop)])
            tachOverThreshPercent[idx-1] = np.sum(isInWindRegion)/(windStop-windStart)
            isFlyingDuringWind = (tachOverThreshPercent[idx-1] >= const.percentageFlying)
            flights[idx-1] = isFlyingDuringWind
            print('Flight: ' + str(isFlyingDuringWind) + ' (' + str(tachOverThreshPercent[idx-1]) + ' percent flight)')
            np.save(saved_flight_fn, flights)

        # these are flight trials (over threshold, determined above)
        if flights[idx-1] & ~(np.shape(angles)[0] < const.numFrames):
            isFlying[idx-1] = 1
        # Partial flight trials are omitted
        # Occasionally acquisition was stopped in the middle of a trial (like at the end of an experiment)
        elif (np.shape(angles)[0] < const.numFrames) | (tachOverThreshPercent[idx-1] >= 0.01):
            isFlying[idx-1] = 0.5
        else: #fully non-flying trials
            isFlying[idx-1] = 0
        flyingPercent[idx-1] = tachOverThreshPercent[idx-1]

        if importAnew:# & ~os.path.isfile(saved_angle_fn):
            np.save(saved_isFlying_fn, isFlying)
            np.save(saved_flyingPercent_fn, flyingPercent)
            np.save(saved_overThresh_fn, overThresh)

    return isFlying, flyingPercent, overThresh


# Compute average baseline and stimulus-evoked antennal responses
# if allowFlight = 0, will exclude any trials from main average during which fly was flying (and create a separate list of flight trials)
# if allowFlight = 1, will includ parital and full flight trials (useful for active movement anlysis and traces)
#  Returns wind, light-activated
# if TEST, will plot that trial (e.g. TEST=2 for trial number 2) or if -1, will plot every trial
def getAntennaTracesAvgs_singleExpt(expt = '2020_11_13_E4', cameraView = 'frontal', importAnew = 0, allowFlight=0, TEST=0, plotInactivate=0):
    #allowFlight=0 #set this to 1 if you want to re-extract/save data including trials with flight (not elegant but works!)
    #TEST = 1
    SAVE_TEST = 1
    exptNotes = get_mat_notes(expt)
    activate = get_activation_trials(expt)
    numTrials = np.shape(exptNotes)[0]
    stimStart = exptNotes.pre_trial_time[0][0]
    stimStop = exptNotes.pre_trial_time[0][0]+exptNotes.trial_time_wind[0][0]
    windStart =stimStart*const.samplerate
    windStop = stimStop*const.samplerate

    scaleBarSize = 10
    scaleWidth = 1
    scaleY = 5
    framerate = const.framerate
    stimStart = const.activateAvgSt
    preLightInd = int(const.preLightOn*framerate)
    postLightInd = int(const.postLightOn*framerate)
    rangeTrace = range(preLightInd, postLightInd-3) #how much of trace to plot (start to onset of wind)
    scaleX = -5 #put vertical scale bar to the right of the traces
    if cameraView == 'frontal':
        yaxis_max = 50
    else:
        yaxis_max = 70
    yaxis_min = -7
    xaxis_min = -10
    xaxis_max = const.lenVideo*const.framerate+const.framerate

    avgBase = np.empty([numTrials,np.shape(const.angPairNames[cameraView])[0]])
    avgBase[:] = np.NaN
    avgLight = np.empty([numTrials,np.shape(const.angPairNames[cameraView])[0]])
    avgLight[:] = np.NaN
    avgWind = np.empty([numTrials,np.shape(const.angPairNames[cameraView])[0]])
    avgWind[:] = np.NaN
    avgBaseFlight = np.empty([numTrials,np.shape(const.angPairNames[cameraView])[0]])
    avgBaseFlight[:] = np.NaN
    avgLightFlight = np.empty([numTrials,np.shape(const.angPairNames[cameraView])[0]])
    avgLightFlight[:] = np.NaN
    avgWindFlight = np.empty([numTrials,np.shape(const.angPairNames[cameraView])[0]])
    avgWindFlight[:] = np.NaN
    flights = np.empty([numTrials], dtype=bool)
    tachOverThreshPercent = np.empty([numTrials])
    # angs_all will contain all traces relative to midline for this experiment
    angs_all = np.empty([numTrials, const.framerate*const.lenVideo,
        np.shape(const.angPairNames[cameraView])[0]])
    bb, aa = generate_butter_filter(samplerate = const.samplerate) #make filter for analyzing tachometer signal

    for idx in range(1,numTrials+1):
        if (plotInactivate==0) | ((plotInactivate==1) & activate[idx-1]) | ((plotInactivate==2) & ~activate[idx-1]):
            saved_angle_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'angles.npy'
            saved_angleFun_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'anglesFun.npy' #supplemental angle - right and left funiculus! (tip-arista base)
            saved_flight_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'flights.npy'
            saved_tach_fn = const.savedDataDirectory+expt+'_'+str(idx)+'_'+cameraView+'_'+'tach.npy'
            tach = exptNotes.tachometer[idx-1]
            if (~importAnew & os.path.isfile(saved_angleFun_fn)): #check if the notes have been loaded and saved previously
                angles = np.load(saved_angle_fn)
                anglesFun = np.load(saved_angleFun_fn)
                flights = np.load(saved_flight_fn)
                tachOverThreshPercent = np.load(saved_tach_fn)
            else: #load notes from scratch (first time analyzing!)
                print('Importing angles for this experiment: '+expt + ' trial '+str(idx))
                angles = get_antenna_angles_adjusted(expt, cameraView, idx);
                np.save(saved_angle_fn, angles)

                tachOverThresh, overThresh = detect_flight(tach,bb,aa);
                isInWindRegion = 1*np.array([(tachOverThresh>=windStart) & (tachOverThresh<=windStop)])
                tachOverThreshPercent[idx-1] = np.sum(isInWindRegion)/(windStop-windStart)
                isFlyingDuringWind = (tachOverThreshPercent[idx-1] >= const.percentageFlying)
                flights[idx-1] = isFlyingDuringWind
                print('Flight: ' + str(isFlyingDuringWind) + ' (' + str(tachOverThreshPercent[idx-1]) + ' percent flight)')
                np.save(saved_flight_fn, flights)
                np.save(saved_tach_fn, tachOverThreshPercent)

            avgIndsBase = np.arange(const.baseAvgSt*const.framerate, const.baseAvgLen*const.framerate)
            avgIndsLight = np.arange(const.activateAvgSt*const.framerate, (const.activateAvgSt+const.activateAvgLen)*const.framerate)
            avgIndsWind = np.arange(const.windAvgSt*const.framerate, (const.windAvgSt+const.windAvgLen)*const.framerate)
            # these are flight trials (over threshold, determined above)
            if flights[idx-1] & ~(np.shape(angles)[0] < const.numFrames) & ~allowFlight:
                print('flight on trial '+str(idx) + ' (' + expt +')')
                avgBaseFlight[idx-1][:] = np.nanmean(angles[avgIndsBase,:],0)
                avgLightFlight[idx-1][:] = np.nanmean(angles[avgIndsLight,:],0)
                avgWindFlight[idx-1][:] = np.nanmean(angles[avgIndsWind,:],0)
                angs_all[idx-1,:,:] = angles
            # Partial flight trials are omitted
            # Occasionally acquisition was stopped in the middle of a trial (like at the end of an experiment)
            elif (np.shape(angles)[0] < const.numFrames) | (tachOverThreshPercent[idx-1] >= 0.01) & ~allowFlight:
                print('partial flight')
                if np.shape(angles)[0] < const.numFrames:
                    print('Video cutoff for experiment '+expt+', trial '+str(idx)+'(numberOfFrames='+ str(np.shape(angles)[0])+') - nan this trial')
                elif TEST != 0:
                    print('partial flight on trial ' + str(idx) + ' (tachOverThreshPercent= '+str(tachOverThreshPercent[idx-1])+ ')')
                nanarr = np.empty(np.shape(avgBase[0][:]))
                nanarr[:] = np.NaN
                nanarr2 = np.empty(np.shape(angs_all[0,:,:]))
                nanarr2[:] = np.NaN
                avgBase[idx-1][:] = nanarr #omitting partial flight trials
                avgLight[idx-1][:] = nanarr
                avgWind[idx-1][:] = nanarr
                angs_all[idx-1,:,:] = nanarr2
            else: #fully non-flying trials
                avgBase[idx-1][:] = np.nanmean(angles[avgIndsBase,:],0)
                avgLight[idx-1][:] = np.nanmean(angles[avgIndsLight,:],0)
                avgWind[idx-1][:] = np.nanmean(angles[avgIndsWind,:],0)
                angs_all[idx-1,:,:] = angles
            #testing (some hard-coding in here!)
            if (idx == TEST) | (TEST == -1):
                windVel = exptNotes.velocity[idx-1]
                windVel = windVel[0] #take values out of brackets
                valveStates = exptNotes.valveState[idx-1]
                windDir = valveStates[0]/2-1 #convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
                windDir = windDir.astype(int)
                fig, axAng = plt.subplots(facecolor=[1,1,1],figsize=(8,8))
                plt.title(expt+', trial '+str(idx)+' vel= '+str(windVel)+ ' dir= '+ const.windDirNames[windDir], color=const.axisColor, loc='left')

                # add vertical scale bar
                axAng.add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
                    facecolor = const.axisColor))
                axAng.text(scaleX-scaleWidth*6, scaleY+scaleBarSize/2,
                    str(scaleBarSize) + const.degree_sign,color=const.axisColor,
                    fontsize=const.fontSize_angPair,horizontalalignment='left',
                    verticalalignment='center')

                #draw light activation stimulus bar
                rectX = int(const.activateStart*const.framerate)
                rectY = yaxis_min#-rectHeight*.05
                rectWid = int(const.activateTime*const.framerate)
                axAng.add_patch(Rectangle((rectX,rectY),rectWid,const.stimBar_height,facecolor = const.color_activateColor))
                axAng.text(rectX+rectWid/2, rectY-const.stimBar_height*2-const.fontSize_stimBar/const.scaleAng_rawTraces/2,
                    str(int(const.activateTime))+' s light on',color=const.color_activateColor,
                    fontsize=const.fontSize_stimBar,horizontalalignment='center')
                #draw wind activation
                rectX = int(const.windStart*const.framerate)
                rectWidWind = int(const.windTime*const.framerate)
                rectY_wind = yaxis_min+1.5
                axAng.text(rectX+rectWidWind/3, rectY_wind+const.stimBar_height*2+const.fontSize_stimBar/const.scaleAng_rawTraces,
                    str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,
                    horizontalalignment='center',color=const.axisColor)
                axAng.add_patch(Rectangle((rectX,rectY_wind),rectWidWind,const.stimBar_height,
                facecolor = const.axisColor))

                framerate = exptNotes.fps[0][0]
                numPreInds = int(exptNotes.pre_trial_time[0]*framerate)
                baselineInds = list(range(0, numPreInds))
                angs = [0,1,2,3]
                for ang in angs:
                    baseline = np.nanmean(angles[baselineInds, ang])
                    shift = ang*const.shiftYTraces  # shifts traces (in y) relative to each other (for ease of viewing)
                    axAng.plot(angles[:, ang]-baseline+shift, color=const.colors_antAngs[ang])
                    agg = angs_all[idx-1,:,ang]
                    axAng.plot(agg-baseline+shift, color=const.colors_antAngs[ang])

                #plot 3rd-2nd
                leftDiff = angles[:, 3]-angles[:, 2]
                baseline = np.nanmean(leftDiff[baselineInds])
                shift = 2.5*const.shiftYTraces  # shifts traces (in y) relative to each other (for ease of viewing)
                axAng.plot(leftDiff-baseline+shift, color='magenta')
                rightDiff = angles[:, 1]-angles[:, 0]
                baseline = np.nanmean(rightDiff[baselineInds])
                shift = 0.5*const.shiftYTraces  # shifts traces (in y) relative to each other (for ease of viewing)
                axAng.plot(rightDiff-baseline+shift, color='cyan')

                # also plot tachometer signal
                axAng.plot(tach[0::int(const.samplerate/const.framerate)+1]+(ang+1)*const.shiftYTraces,color=const.flight_color)

                plt.pause(0.001)
                #plt.show(block=False)
                plt.show()

                if SAVE_TEST:
                    today = date.today()
                    dateStr = today.strftime("%Y_%m_%d")
                    figPath = const.savedFigureDirectory+str(dateStr)+'/'
                    if not os.path.isdir(figPath):
                        os.mkdir(figPath)
                    savepath = figPath + expt+ '_trial_'+str(idx)+'_allTraces'+'.pdf'
                    fig.savefig(savepath, facecolor=fig.get_facecolor())
                    savepath = figPath + expt+ '_trial_'+str(idx)+'_allTraces'+ '.png'
                    fig.savefig(savepath, facecolor=fig.get_facecolor())


    return angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight


# Compute cross-fly average traces across all velocities and directions
def getAntennaTracesAvgs_crossExpt(expt='74C10_activate', cameraView='frontal', importAnew = 0,allowFlight=0,plotInactivate=0):
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)
    # gather single-fly antennal angle averages
    for noteInd, notes in enumerate(allExpts):
        flyExpt = notes['date']
        exptNotes = get_mat_notes(flyExpt)
        windVel = exptNotes.velocity[:]
        windVel = windVel.str.get(0)  # take values out of brackets
        valveStates = exptNotes.valveState[:]
        windDir = valveStates.str.get(0)/2-1  # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
        windDir = windDir.astype(int)
        windDirs = np.unique(windDir)
        speeds = np.unique(windVel)
        uniqueValveStates = np.unique(valveStates.str.get(0))
        antInds = [0,1,2,3]

        [angs_all, _, _, _, _, _, _] = getAntennaTracesAvgs_singleExpt(flyExpt, cameraView, importAnew, allowFlight=allowFlight, TEST=0,plotInactivate=plotInactivate)

        angsAll = detect_tracking_errors_raw_trace(exptNotes, expt, angs_all)
        if noteInd == 0:  # initialize main data structures
            tracesAllFlies = np.empty([np.shape(allExpts)[0],np.shape(angs_all)[1],
                                       np.shape(angs_all)[2],np.shape(uniqueValveStates)[0],
                                       np.shape(speeds)[0]])
        # gather averages across all trials for this fly of same type (for each velocity-direction combo)
        for dirIdx, state in enumerate(uniqueValveStates):
            dir = uniqueValveStates[dirIdx]
            thisDirInds = np.squeeze(np.where(windDir==dirIdx))
            for velIdx, sp in enumerate(speeds):
                thisVelInds = np.squeeze(np.where(windVel==sp))
                dirVelInds = [value for value in thisDirInds if value in thisVelInds]
                for antInd in antInds:
                    meanFlyTrace = np.nanmean(angs_all[dirVelInds, :, antInd], axis=0)
                    tracesAllFlies[noteInd,:,antInd,dirIdx,velIdx] = meanFlyTrace

    return tracesAllFlies

# Compute and return average wind responses across directions, for each get_velocity_tuning_averages
#  For one experiment
#PLOT_THIRD_ONLY - if 1, will return deflections for third segment. Default is to subtract second from third (for passive antennal deflection estimation)
def get_wind_vel_tuning_singleFly(expt='2020_11_13_E4', cameraView='frontal', importAnew = 0, plotInactivate=0,PLOT_THIRD_ONLY = 0,allowFlight=0):
    print('allowflight='+str(allowFlight)+' on line 708')
    [angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight] = getAntennaTracesAvgs_singleExpt(expt, cameraView, importAnew=importAnew,allowFlight=allowFlight,plotInactivate=plotInactivate)
    exptNotes = get_mat_notes(expt)
    windVel = exptNotes.velocity[:]
    windVel = windVel.str.get(0) #take values out of brackets
    valveStates = exptNotes.valveState[:]
    windDir = valveStates.str.get(0)/2-1 #convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)
    windDirs = np.unique(windDir)
    speeds = np.unique(windVel)
    uniqueValveStates = np.unique(valveStates.str.get(0))
    windRespAllTrials_L        = np.empty([np.shape(avgWind)[1],np.shape(windDirs)[0]]) #store cross-trial average in here
    windRespAllTrials_R        = np.empty([np.shape(avgWind)[1],np.shape(windDirs)[0]]) #store cross-trial average in here
    windRespAllTrials_L_Flight = np.empty([np.shape(avgWind)[1],np.shape(windDirs)[0]]) #store cross-trial average in here
    windRespAllTrials_R_Flight = np.empty([np.shape(avgWind)[1],np.shape(windDirs)[0]]) #store cross-trial average in here

    for velInd in range(np.shape(speeds)[0]):
        respInds = np.where(windVel == speeds[velInd])[0] # indices for speed of interest
        if PLOT_THIRD_ONLY == 1:
            base = avgLight[respInds,const.rightAristaInd] #baseline (pre-wind, during activation)
            rightWindResp = avgWind[respInds, const.rightAristaInd]-base
            base = avgLight[respInds,const.leftAristaInd] #baseline (pre-wind, during activation)
            leftWindResp = avgWind[respInds, const.leftAristaInd]-base

            baseFlight = avgLightFlight[respInds,const.rightAristaInd] #baseline (pre-wind, during activation)
            rightWindRespFlight = avgWindFlight[respInds, const.rightAristaInd]-baseFlight
            baseFlight = avgLightFlight[respInds,const.leftAristaInd] #baseline (pre-wind, during activation)
            leftWindRespFlight = avgWindFlight[respInds, const.leftAristaInd]-baseFlight
        else: #take THIRD-SECOND segment values here! (Previously: quantified just movements of third; subtract second for full passive difference)
            baseArista = avgLight[respInds,const.rightAristaInd] #baseline third seg (pre-wind, during activation)
            rightWindRespArista = avgWind[respInds, const.rightAristaInd]-baseArista #third
            baseSecond = avgLight[respInds,const.rightSecondInd] #baseline second seg (pre-wind, during activation)
            rightWindRespSecond = avgWind[respInds, const.rightSecondInd]-baseSecond #second
            rightWindResp = rightWindRespArista-rightWindRespSecond

            baseArista = avgLight[respInds,const.leftAristaInd] #baseline (pre-wind, during activation)
            leftWindRespArista = avgWind[respInds, const.leftAristaInd]-baseArista
            baseSecond = avgLight[respInds,const.leftSecondInd] #baseline (pre-wind, during activation)
            leftWindRespSecond = avgWind[respInds, const.leftSecondInd]-baseSecond
            leftWindResp = leftWindRespArista-leftWindRespSecond

            baseFlightArista = avgLightFlight[respInds,const.rightAristaInd] #baseline (pre-wind, during activation)
            rightWindRespFlightArista = avgWindFlight[respInds, const.rightAristaInd]-baseFlightArista
            baseFlightSecond = avgLightFlight[respInds,const.rightSecondInd] #baseline (pre-wind, during activation)
            rightWindRespFlightSecond = avgWindFlight[respInds, const.rightSecondInd]-baseFlightSecond
            rightWindRespFlight = rightWindRespFlightArista-rightWindRespFlightSecond

            baseFlightArista = avgLightFlight[respInds,const.leftAristaInd] #baseline (pre-wind, during activation)
            leftWindRespFlightArista = avgWindFlight[respInds, const.leftAristaInd]-baseFlightArista
            baseFlightSecond = avgLightFlight[respInds,const.leftSecondInd] #baseline (pre-wind, during activation)
            leftWindRespFlightSecond = avgWindFlight[respInds, const.leftSecondInd]-baseFlightSecond
            leftWindRespFlight = leftWindRespFlightArista-leftWindRespFlightSecond

        for ii in range(np.shape(windDirs)[0]):
            windRespAllTrials_R[velInd][ii] = np.nanmean(rightWindResp[np.where(windDir[respInds] == windDirs[ii])])
            windRespAllTrials_L[velInd][ii] = np.nanmean(leftWindResp[np.where(windDir[respInds] == windDirs[ii])])

            windRespAllTrials_R_Flight[velInd][ii] = np.nanmean(rightWindRespFlight[np.where(windDir[respInds] == windDirs[ii])])
            windRespAllTrials_L_Flight[velInd][ii] = np.nanmean(leftWindRespFlight[np.where(windDir[respInds] == windDirs[ii])])

    return windRespAllTrials_R, windRespAllTrials_L, angs_all, avgBase, avgLight, avgWind, windRespAllTrials_R_Flight, windRespAllTrials_L_Flight, avgBaseFlight, avgLightFlight, avgWindFlight

# Acquire cross-fly averages for direction and velocity tuning
#  for a given genotype/experiment set
def get_wind_vel_tuning_crossFly(expt='74C10_activate', cameraView='frontal', importAnew = 0,plotInactivate=0,allowFlight=0):
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)
    # NxMxP array, where N = # flies (variable, ~10), M = # velocities (4), P = # directions (5)
    windRespAllTrials_L = np.empty([np.shape(allExpts)[0],const.numVel,const.numDirs])
    windRespAllTrials_R = np.empty([np.shape(allExpts)[0],const.numVel,const.numDirs])

    windRespAllTrials_L_Flight = np.empty([np.shape(allExpts)[0],const.numVel,const.numDirs])
    windRespAllTrials_R_Flight = np.empty([np.shape(allExpts)[0],const.numVel,const.numDirs])

    # Acquire list of experiments for this genotype
    for noteInd, notes in enumerate(allExpts):
        flyExpt = notes['date']
        # get tuning responses for single fly
        [windIndv_R, windIndv_L, angles, base, light, wind, windIndv_R_Flight,
         windIndv_L_Flight, baseFlight, lightFlight, windFlight] = get_wind_vel_tuning_singleFly(flyExpt, cameraView, importAnew,allowFlight=allowFlight,plotInactivate=plotInactivate)
        windRespAllTrials_R[noteInd][:][:] = windIndv_R
        windRespAllTrials_L[noteInd][:][:] = windIndv_L

        windRespAllTrials_R_Flight[noteInd][:][:] = windIndv_R_Flight
        windRespAllTrials_L_Flight[noteInd][:][:] = windIndv_L_Flight

    return windRespAllTrials_R, windRespAllTrials_L, windRespAllTrials_R_Flight, windRespAllTrials_L_Flight


# return difference between right and left antennal deflections
#  currently returns difference between R and L for 200 cm/s
def get_difference_cross_fly(expt='CS_activate',cameraView='frontal',importAnew=0,plotInactivate=0,velToQuant=3):
    if 'inactivate' in expt:
        velToQuant = 1 #200cm/s

    if plotInactivate ==1:
        extraText = '_withLight_'
    elif plotInactivate == 2:
        extraText = '_noLight_'
    else:
        extraText = ''
    windResp_R_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'windResp_R.npy'
    windResp_L_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'windResp_L.npy'
    windResp_R_Flight_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'windResp_R_Flight.npy'
    windResp_L_Flight_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'windResp_L_Flight.npy'

    if (importAnew==1) | (os.path.isfile(windResp_R_fn)==False):
        print('Importing R-L tuning data')
        [windResp_R, windResp_L, windResp_R_Flight, windResp_L_Flight] = get_wind_vel_tuning_crossFly(expt, cameraView, importAnew,plotInactivate=plotInactivate)
        np.save(windResp_R_fn, windResp_R)
        np.save(windResp_L_fn, windResp_L)
        np.save(windResp_R_Flight_fn, windResp_R_Flight)
        np.save(windResp_L_Flight_fn, windResp_L_Flight)
    else:
        windResp_R = np.load(windResp_R_fn)
        windResp_L = np.load(windResp_L_fn)
        windResp_R_Flight = np.load(windResp_R_fn)
        windResp_L_Flight = np.load(windResp_L_Flight_fn)

    RminL_200cm = np.squeeze(windResp_R[:,velToQuant,:]-windResp_L[:,velToQuant,:])

    return RminL_200cm


# Plot wind direction tuning for a single fly or across a genotype, e.g. 'CS_activate'
def plot_wind_vel_tuning(expt='2020_11_13_E4', cameraView='frontal', errorBar = 0, saveTransparent=True,plotBothAntenna = 1, importAnew = 0, allowFlight = 0, savefig = 0,plotInactivate=0):
    indicateSex = 0 # if 1, will draw red outline around data points that are from female flies
    flight=0 #if 1, returns purefly flight data (if 0, allowFlight determines whether data plotted is nonflying only, or nonlying+nonflying trials)
    rminl_allTrials_fn = ''
    SAVE_STRUCTS = 0 #if 1, will save single-fly RminL data to a .npy file (can combine with setting allowFlight=1 in getAntennaTracesAvgs_singleExpt if you want to save with flight data)
    if plotInactivate ==1:
        extraText = '_withLight_'
    elif plotInactivate == 2:
        extraText = '_noLight_'
    else:
        extraText = '_'
    import loadNotes_mnOptoVelocityTuning as ln
    # plot across all flies for a given experiment
    try:
        allExpts = eval('ln.notes_'+expt)
        fly = 0
        plot_single_fly = 0
        exptNotes = get_mat_notes(allExpts[0]['date']) # all experiments are of the same format, first entry has all necessary details
    except: # plot tuning from a single fly
        plot_single_fly = 1
        exptNotes = get_mat_notes(expt)

    jitterSft = 0.25
    fig, axAng = plt.subplots(1,2,facecolor=const.figColor,figsize=(12,8))
    windVel = exptNotes.velocity[:]
    windVel = windVel.str.get(0) # take values out of brackets
    valveStates = exptNotes.valveState[:]
    windDir = valveStates.str.get(0)/2-1 # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)
    windDirs = np.unique(windDir)
    speeds = np.unique(windVel)
    uniqueValveStates = np.unique(valveStates.str.get(0))
    # if plotting a single fly:
    if plot_single_fly:
        [windRespAllTrials_R, windRespAllTrials_L, angs_all, avgBase, avgLight, avgWind,
            windRespAllTrials_R_Flight, windRespAllTrials_L_Flight, avgBaseFlight, avgLightFlight, avgWindFlight] = get_wind_vel_tuning_singleFly(expt, cameraView, importAnew=importAnew,allowFlight=allowFlight,PLOT_THIRD_ONLY=0,plotInactivate=plotInactivate)
        if flight:
            windRespAllTrials_R = windRespAllTrials_R_Flight
            windRespAllTrials_L = windRespAllTrials_L_Flight
        else:
            windRespCrossFlies_R = windRespAllTrials_R
            windRespCrossFlies_L = windRespAllTrials_L
    # if plotting across flies for a genotype:
    else:
        [windRespAllTrials_R, windRespAllTrials_L, windRespAllTrials_R_Flight, windRespAllTrials_L_Flight] = get_wind_vel_tuning_crossFly(expt, cameraView, importAnew=importAnew,plotInactivate=plotInactivate)
        if flight:
            windRespCrossFlies_R = np.nanmean(windRespAllTrials_R_Flight,0)
            windRespCrossFlies_L = np.nanmean(windRespAllTrials_L_Flight,0)
        else:
            windRespCrossFlies_R = np.nanmean(windRespAllTrials_R,0)
            windRespCrossFlies_L = np.nanmean(windRespAllTrials_L,0)

    topLegendLeft = const.yaxis_max_tuning-1
    topLegendRight = const.yaxis_max_tuning-const.yaxis_max_tuning*0.15-1
    # plot wind tuning for this experiment (for all speeds)
    # wind averages are plotted relative activation baseline
    numFlies = np.shape(windRespAllTrials_R)[0]
    windDirsRepeat = np.tile(windDirs,[numFlies,1])
    #for velInd in range(const.numVel):
    for velInd in range(np.shape(speeds)[0]):
        # plot individual trial/fly responses
        if plot_single_fly:
            respInds = np.where(windVel == speeds[velInd])[0] # indices for speed of interest
            jitter = jitterSft*np.random.rand(len(windDir[respInds]))-jitterSft/2 # add some random jitter to x-placement of single trial responses
            xxR = windDir[respInds]+jitter
            jitter = jitterSft*np.random.rand(len(windDir[respInds]))-jitterSft/2 # add some random jitter to x-placement of single trial responses
            xxL = windDir[respInds]+jitter

            if flight:
                base = avgLightFlight[respInds,const.rightAristaInd] # baseline (pre-wind, during activation)
                rightWindResp = avgWindFlight[respInds, const.rightAristaInd]-base

                base = avgLightFlight[respInds,const.leftAristaInd] # baseline (pre-wind, during activation)
                leftWindResp = avgWindFlight[respInds, const.leftAristaInd]-base
            else:
                baseArista = avgLight[respInds,const.rightAristaInd] #baseline third seg (pre-wind, during activation)
                rightWindRespArista = avgWind[respInds, const.rightAristaInd]-baseArista #third
                baseSecond = avgLight[respInds,const.rightSecondInd] #baseline second seg (pre-wind, during activation)
                rightWindRespSecond = avgWind[respInds, const.rightSecondInd]-baseSecond #second
                rightWindResp = rightWindRespArista-rightWindRespSecond

                baseArista = avgLight[respInds,const.leftAristaInd] #baseline (pre-wind, during activation)
                leftWindRespArista = avgWind[respInds, const.leftAristaInd]-baseArista
                baseSecond = avgLight[respInds,const.leftSecondInd] #baseline (pre-wind, during activation)
                leftWindRespSecond = avgWind[respInds, const.leftSecondInd]-baseSecond
                leftWindResp = leftWindRespArista-leftWindRespSecond

        # plot cross-fly averages
        else:
            if flight:
                rightWindResp = windRespAllTrials_R_Flight[:,velInd,:]
                leftWindResp = windRespAllTrials_L_Flight[:,velInd,:]
            else:
                rightWindResp = windRespAllTrials_R[:,velInd,:]
                leftWindResp = windRespAllTrials_L[:,velInd,:]
            xxR = windDirsRepeat+jitterSft*np.random.rand(np.shape(windDirsRepeat)[0], np.shape(windDirsRepeat)[1])-jitterSft/2
            xxL = windDirsRepeat+jitterSft*np.random.rand(np.shape(windDirsRepeat)[0], np.shape(windDirsRepeat)[1])-jitterSft/2

        axAng[0].plot(xxR, rightWindResp, color=const.colors_velocity_right[velInd],
            marker='.', linestyle='None', markerSize = const.markerTuningIndv)#,
            #alpha=const.transparencyTuningPoints)
        if plotBothAntenna:
            axAng[0].plot(xxL, leftWindResp, color=const.colors_velocity_left[velInd],
                marker='.', linestyle='None', markerSize = const.markerTuningIndv)#,
                #alpha=const.transparencyTuningPoints)
        if plot_single_fly: #do this later for cross-fly
            RminL_singleTrials = rightWindResp-leftWindResp
            windDirSingleTrials = windDir[respInds]
            axAng[1].plot(xxR, RminL_singleTrials, marker='.', linestyle='None', markerSize = const.markerTuningIndv, color=const.colors_velocity_RminL[velInd])

            if SAVE_STRUCTS==1:
                # get flight data and save with direction and steady state response
                isFlying, flyingPercent, overThresh = getFlightsSingleExpt(expt )
                flights = isFlying[respInds]
                RminL_singleTrialData = [RminL_singleTrials.tolist(),windDirSingleTrials.tolist(), flights.tolist()]
                rminl_allTrials_singleFly_fn = const.savedDataDirectoryForSingleTrials+expt+'_'+cameraView+'_vel'+str(velInd)+extraText+'RminL_allTrials.npy'
                print('saving data here: ' + rminl_allTrials_singleFly_fn)
                np.save(rminl_allTrials_singleFly_fn,RminL_singleTrialData)
                mat_fn = rminl_allTrials_singleFly_fn[:-4]+'.mat'
                savemat(mat_fn,{'data':RminL_singleTrialData})
                print('saving data here in .mat: ' + mat_fn)
        # plot average single/cross-fly responses
        if plot_single_fly:
            axAng[0].plot(windDirs, windRespAllTrials_R[velInd][:], marker='.', markerSize = const.markerTuningAvg, color=const.colors_velocity_right[velInd])
            if plotBothAntenna:
                axAng[0].plot(windDirs, windRespAllTrials_L[velInd][:], marker='.', markerSize = const.markerTuningAvg, color=const.colors_velocity_left[velInd])
        else:
            axAng[0].plot(windDirs, windRespCrossFlies_R[velInd][:], marker='.', markerSize = const.markerTuningAvg, color=const.colors_velocity_right[velInd])
            if plotBothAntenna:
                axAng[0].plot(windDirs, windRespCrossFlies_L[velInd][:], marker='.', markerSize = const.markerTuningAvg, color=const.colors_velocity_left[velInd])

        # add legend indicating left/right antenna and speeds
        shiftAmt = 22/const.yaxis_max_tuning

        if plotBothAntenna:
            axAng[0].plot([4, 4.5],[topLegendLeft-velInd*shiftAmt, topLegendLeft-velInd*shiftAmt], marker='None', markerSize = const.markerTuningAvg, color=const.colors_velocity_left[velInd])
            axAng[0].text(4.6,topLegendLeft-velInd*shiftAmt, str(speeds[velInd])+' cm/s', color=const.colors_velocity_left[velInd])
        axAng[0].plot([4, 4.5],[topLegendRight-velInd*shiftAmt-shiftAmt*3, topLegendRight-velInd*shiftAmt-shiftAmt*3], marker='None', markerSize = const.markerTuningAvg, color=const.colors_velocity_right[velInd])
        axAng[0].text(4.6,topLegendRight-velInd*shiftAmt-shiftAmt*3, str(speeds[velInd])+' cm/s', color=const.colors_velocity_right[velInd])

        # Plot R-L antenna curve to the right of individual antenna curves
        if plot_single_fly:
            RminL = windRespAllTrials_R[velInd][:]-windRespAllTrials_L[velInd][:]
            axAng[1].plot(windDirs, RminL, marker='.', markerSize = const.markerTuningAvg, color=const.colors_velocity_RminL[velInd])
        else:
            # plot all trials for R-L average tuning
            RminL_avg = windRespCrossFlies_R[velInd][:]-windRespCrossFlies_L[velInd][:]
            RminL_allTrials = np.squeeze(windRespAllTrials_R[:,velInd,:]-windRespAllTrials_L[:,velInd,:])
            windDirsRepeat = np.tile(windDirs,[numFlies,1])
            xxR = windDirsRepeat+jitterSft*np.random.rand(np.shape(windDirsRepeat)[0], np.shape(windDirsRepeat)[1])-jitterSft/2

            if errorBar==1: # plot R-L error (very hard to see when overlapping)
                error = np.nanstd(RminL_allTrials) # standard deviation
                axAng[1].errorbar(windDirs, RminL_avg, error, alpha=const.transparencyTuningPoints, markerSize = const.markerTuningIndv, color=const.colors_velocity_RminL[velInd])
            elif errorBar == 2:
                error = np.nanstd(RminL_allTrials)/np.sqrt(np.shape(RminL_allTrials)[0]) # std/sqrt(#flies)= standard error
                axAng[1].errorbar(windDirs, RminL_avg, error, alpha=const.transparencyTuningPoints, markerSize = const.markerTuningIndv, color=const.colors_velocity_RminL[velInd])
            else: # plot individual R-L trials/flies
                if indicateSex: # indicate which flies were female
                    for flyNum in range(numFlies):
                        if allExpts[flyNum]['sex'] == 'female':
                            flyEdgeColor = 'yellow'
                        else: flyEdgeColor = 'None'
                        jitter = jitterSft*np.random.rand(len(windDirs))-jitterSft/2
                        axAng[1].plot(windDirs+jitter, RminL_allTrials[flyNum][:], marker='.', linestyle = 'None',
                            markerSize = const.markerTuningIndv,
                            color=const.colors_velocity_RminL[velInd],markeredgecolor=flyEdgeColor)#alpha=const.transparencyTuningPoints,
                else:
                    axAng[1].plot(xxR, RminL_allTrials, marker='.', linestyle = 'None',
                        markerSize = const.markerTuningIndv, color=const.colors_velocity_RminL[velInd])#    , alpha=const.transparencyTuningPoints, )

            axAng[1].plot(windDirs, RminL_avg, marker='.', markerSize = const.markerTuningAvg, color=const.colors_velocity_RminL[velInd])

        #add legend indicating R-L antenna and speeds
        axAng[1].plot([4, 4.5],[topLegendRight-velInd*shiftAmt-shiftAmt*3, topLegendRight-velInd*shiftAmt-shiftAmt*3], marker='None', markerSize = const.markerTuningAvg, color=const.colors_velocity_RminL[velInd])
        axAng[1].text(4.6,topLegendRight-velInd*shiftAmt-shiftAmt*3, str(speeds[velInd])+' cm/s', color=const.colors_velocity_RminL[velInd])

    if 'inactivate' in expt:
        velToQuant = 1
    else:
        velToQuant = 3
    if plot_single_fly:
        RminL_200cm = np.squeeze(windRespAllTrials_R[velToQuant,:]-windRespAllTrials_L[velToQuant,:])
    else:
        RminL_200cm = np.squeeze(windRespAllTrials_R[:,velToQuant,:]-windRespAllTrials_L[:,velToQuant,:])

    if plotBothAntenna:
        axAng[0].text(4,topLegendLeft+shiftAmt*0.9, 'Left antenna', color='gray')
    axAng[0].text(4,topLegendRight-shiftAmt*2, 'Right antenna', color='gray')
    axAng[1].text(4,topLegendRight-shiftAmt*2, 'Right-Left antenna', color='gray')

    # configure the axes, title, etc.
    if plot_single_fly:
        fig.suptitle('Wind and velocity tuning (single experiment) \n'+cameraView+' '+exptNotes.notes[0] + '\n' + exptNotes.date[0] + '_E' + str(exptNotes.expNumber[0][0]), fontsize=const.fsize_title_tuning,color=const.axisColor)
    else:
        fig.suptitle('Wind and velocity tuning (across flies) \n'+cameraView+' '+expt, fontsize=const.fsize_title_tuning,color=const.axisColor)
    for ii in range(2):
        axAng[ii].set_facecolor(const.figColor)
        axAng[ii].spines['right'].set_visible(False)
        axAng[ii].spines['top'].set_visible(False)
        axAng[ii].spines['left'].set_color(const.axisColor)
        axAng[ii].spines['bottom'].set_color(const.axisColor)
        axAng[ii].tick_params(direction='in', length=5, width=0.5)
        axAng[ii].tick_params(axis='y',colors=const.axisColor)
        #configure the y-axis
        axAng[ii].set_ylabel('Antennal deflection (deg)',color=const.axisColor)
        if 'inactivate' in expt:
            axAng[ii].set_ylim(-30, 35)
        else:
            axAng[ii].set_ylim(const.yaxis_min_tuning, const.yaxis_max_tuning)
        #configure the x-axis
        axAng[ii].set_xticks(np.arange(5))
        axAng[ii].tick_params(axis='x',colors=const.axisColor)
        axAng[ii].set_xticklabels(const.windDirLabels)
        axAng[ii].spines['bottom'].set_bounds(0, 4) #do not extend x-axis line beyond ticks
        axAng[ii].set_xlabel('Wind direction (deg)',color=const.axisColor)

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        if plotInactivate ==1:
            extraText = '_withLight_'
        elif plotInactivate == 2:
            extraText = '_noLight_'
        else:
            extraText = ''
        if plotBothAntenna:
            fig.savefig(figPath+'/directionVelocityTuning_'+expt+'_'+cameraView+extraText+'.png')#, transparent=saveTransparent)
            fig.savefig(figPath+'/directionVelocityTuning_'+expt+'_'+cameraView+extraText+'.pdf')#, transparent=saveTransparent)
        else:
            fig.savefig(figPath+'/'+expt+'_'+cameraView+extraText+'singleAnt.png', facecolor=fig.get_facecolor())
            fig.savefig(figPath+'/'+expt+'_'+cameraView+extraText+'singleAnt.pdf', facecolor=fig.get_facecolor())

    if SAVE_STRUCTS:
        plt.close()
    else:
        plt.show(block=False)
        plt.pause(0.001)
        plt.show()

    return RminL_200cm

#hard-coded saving of direction tuning for 200 cm/s
def save_wind_vel_tuning_cross_fly(expt='18D07_inactivate',plotInactivate=1):
    cameraView = 'frontal'
    importAnew = 0 #this will extra angles from scratch (takes forever, only do if we change some base property of this)
    if plotInactivate ==1:
        extraText = '_withLight_'
    elif plotInactivate == 2:
        extraText = '_noLight_'
    else:
        extraText = ''
    RminL_avg_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'RminL_avg.npy'
    RminL_allTrials_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'RminL_allTrials.npy'

    [windRespAllTrials_R, windRespAllTrials_L, windRespAllTrials_R_Flight, windRespAllTrials_L_Flight] = get_wind_vel_tuning_crossFly(expt, cameraView, importAnew,plotInactivate=plotInactivate)
    windRespCrossFlies_R = np.nanmean(windRespAllTrials_R,0)
    windRespCrossFlies_L = np.nanmean(windRespAllTrials_L,0)
    if 'inactivate' in expt:
        velInd = 1
    else: velInd = 3
    RminL_avg = windRespCrossFlies_R[velInd][:]-windRespCrossFlies_L[velInd][:]
    RminL_allTrials = np.squeeze(windRespAllTrials_R[:,velInd,:]-windRespAllTrials_L[:,velInd,:])

    np.save(RminL_avg_fn, RminL_avg)
    np.save(RminL_allTrials_fn, RminL_allTrials)


def plot_wind_vel_tuning_activation_inactivation(saveNewStructs=0):
    importAnew = 0
    cameraView = 'frontal'

    fig, ax = plt.subplots(2,4,facecolor=const.figColor,figsize=(12,8))

    activationExpts = ['CS_activate','18D07_activate','91F02_activate','74C10_activate']
    inactivationExpts = ['emptyGAL4_inactivate','18D07_inactivate','91F02_inactivate']
    expts = activationExpts+inactivationExpts
    inactivation = [0,0,0,0,1,1,1]
    subplots = [[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2]]
    velInds = [3,3,3,3,1,1,1]
    ylimAll = [-20,20]

    for ii in range(np.shape(inactivation)[0]):
        expt = expts[ii]
        plotInactivate = inactivation[ii]
        velInd = velInds[ii]
        saved_avgTuningR_fn = const.savedDataDirectory+expt+'_'+'avgDirectionTuningR.npy'
        saved_avgTuningL_fn = const.savedDataDirectory+expt+'_'+'avgDirectionTuningL.npy'
        saved_avgTuningR_lightOff_fn = const.savedDataDirectory+expt+'_'+'avgDirectionTuningR_lightOff.npy'
        saved_avgTuningL_lightOff_fn = const.savedDataDirectory+expt+'_'+'avgDirectionTuningL_lightOff.npy'
        #all fly averages
        saved_avgTuningRAllTrials_fn = const.savedDataDirectory+expt+'_'+'avgDirectionTuningRAllTrials.npy'
        saved_avgTuningLAllTrials_fn = const.savedDataDirectory+expt+'_'+'avgDirectionTuningLAllTrials.npy'
        saved_avgTuningRAllTrials_lightOff_fn = const.savedDataDirectory+expt+'_'+'avgDirectionTuningRAllTrials_lightOff.npy'
        saved_avgTuningLAllTrials_lightOff_fn = const.savedDataDirectory+expt+'_'+'avgDirectionTuningLAllTrials_lightOff.npy'

        if (saveNewStructs==0) & (os.path.isfile(saved_avgTuningR_fn)==True):
            # load previously saved active movement data
            avgTuningR = np.load(saved_avgTuningR_fn)
            avgTuningL = np.load(saved_avgTuningL_fn)
            if plotInactivate == 1: #also grab light off data
                avgTuningR_lightOff = np.load(saved_avgTuningR_lightOff_fn)
                avgTuningL_lightOff = np.load(saved_avgTuningL_lightOff_fn)
        else:
            print('loading new cross-trial tuning data! ' + expt)
            [windRespAllTrials_R, windRespAllTrials_L, _, _] = get_wind_vel_tuning_crossFly(expt, cameraView, importAnew,plotInactivate=plotInactivate)
            avgTuningR = np.nanmean(windRespAllTrials_R,0)
            avgTuningL = np.nanmean(windRespAllTrials_L,0)
            np.save(saved_avgTuningR_fn, avgTuningR)
            np.save(saved_avgTuningL_fn, avgTuningL)
            np.save(saved_avgTuningRAllTrials_fn, windRespAllTrials_R)
            np.save(saved_avgTuningLAllTrials_fn, windRespAllTrials_L)
            if plotInactivate == 1: #also grab light off data
                [windRespAllTrials_R, windRespAllTrials_L, _, _] = get_wind_vel_tuning_crossFly(expt, cameraView, importAnew,plotInactivate=2)
                avgTuningR_lightOff = np.nanmean(windRespAllTrials_R,0)
                avgTuningL_lightOff = np.nanmean(windRespAllTrials_L,0)
                np.save(saved_avgTuningR_lightOff_fn, avgTuningR_lightOff)
                np.save(saved_avgTuningL_lightOff_fn, avgTuningL_lightOff)
                np.save(saved_avgTuningRAllTrials_lightOff_fn, windRespAllTrials_R)
                np.save(saved_avgTuningLAllTrials_lightOff_fn, windRespAllTrials_L)

        RminL_avg = avgTuningR[velInd][:]-avgTuningL[velInd][:]
        ax[subplots[ii][0],subplots[ii][1]].plot(RminL_avg,color='blue')
        if plotInactivate == 1:
            RminL_lightOff_avg = avgTuningR_lightOff[velInd][:]-avgTuningL_lightOff[velInd][:]
            ax[subplots[ii][0],subplots[ii][1]].plot(RminL_lightOff_avg,color='black')
        if ii == 4: #emptyGAL4
            ax[1,3].plot(RminL_avg,color=const.med_gray)
        elif ii == 5: #18D07
            ax[1,3].plot(RminL_avg,color=const.magenta)
        elif ii == 6: #91F02
            ax[1,3].plot(RminL_avg,color=const.teal)
            ax[1,3].set_ylim(ylimAll)
            ax[1,3].set_title('inactivation overlay')

        ax[subplots[ii][0],subplots[ii][1]].set_title(expt)
        ax[subplots[ii][0],subplots[ii][1]].set_ylim(ylimAll)

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()


# todo: write function to compute differences separately from plot_wind_vel_tuning
# (which should also inherit these values from another function instead of
# computing within)
def get_gain_tuning_quant(expt='74C10_activate', cameraView='frontal', plotInactivate = 0, velToQuant=3, savefig=0,importAnew=0):
    RminL = get_difference_cross_fly(expt,cameraView,importAnew=importAnew,plotInactivate=plotInactivate,velToQuant=velToQuant)
    indNeg45 = 1
    indPos45 = 3
    diff45 = RminL[:,indNeg45]-RminL[:,indPos45]
    return RminL, diff45

def round_to_significant_digit(xx,sig=2):
    return round(xx,sig-int(floor(log10(abs(xx))))-1)

# plot different between +45 and -45 for R-L tuning curves across inactivation genotypes
# Plots this for just 200 cm/s in this version (see plot_gain_tuning_quant for cross-velocity comparison)
def plot_gain_tuning_quant_200cm(cameraView='frontal',savefig=0):

    RminL_CS, diff45_CS = get_gain_tuning_quant(expt='CS_activate',cameraView=cameraView,velToQuant=3,  savefig=0)
    diff45_CS = diff45_CS[~np.isnan(diff45_CS)]

    RminL_91F02, diff45_91F02 = get_gain_tuning_quant(expt='91F02_activate', cameraView=cameraView,velToQuant=3,  savefig=0)
    diff45_91F02 = diff45_91F02[~np.isnan(diff45_91F02)]

    RminL_18D07, diff45_18D07 = get_gain_tuning_quant(expt='18D07_activate', cameraView=cameraView,velToQuant=3,  savefig=0)
    diff45_18D07 = diff45_18D07[~np.isnan(diff45_18D07)]

    RminL_74C10, diff45_74C10 = get_gain_tuning_quant(expt='74C10_activate', cameraView=cameraView, velToQuant=3, savefig=0)
    diff45_74C10 = diff45_74C10[~np.isnan(diff45_74C10)]

    data = [diff45_CS, diff45_18D07, diff45_91F02, diff45_74C10]
    xlabels = ['control','MN line 1', 'MN line 2', 'muscle line']
    # Plot the data
    fig, ax = plt.subplots(1,1,facecolor=const.figColor,figsize=(4,5))
    plt.suptitle('gain (-45° - +45°)',
     color=const.axisColor,fontsize=const.fontSize_angPair)

    #plot all the data (points w/ avg)
    jitterSft = 0.25
    ones = np.ones(np.shape(diff45_CS)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_CS)[0])-jitterSft/2
    ax.plot(1*ones+jitter, diff45_CS,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(1, np.nanmean(diff45_CS),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(diff45_18D07)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_18D07)[0])-jitterSft/2
    ax.plot(2*ones+jitter, diff45_18D07,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(2, np.nanmean(diff45_18D07),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(diff45_91F02)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_91F02)[0])-jitterSft/2
    ax.plot(3*ones+jitter, diff45_91F02,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(3, np.nanmean(diff45_91F02),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(diff45_74C10)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_74C10)[0])-jitterSft/2
    ax.plot(4*ones+jitter, diff45_74C10,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(4, np.nanmean(diff45_74C10),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    # plot result of ttest between groups (indicate significane level)
    result2 = stats.ttest_ind(diff45_CS, diff45_91F02)
    result1 = stats.ttest_ind(diff45_CS, diff45_18D07)
    result3 = stats.ttest_ind(diff45_CS, diff45_74C10)
    pVals = [result1.pvalue, result2.pvalue, result3.pvalue]
    ax.text(4.5, 34, 'CS vs. 18D07: ' +str(round_to_significant_digit(result1.pvalue,2)), color='pink')
    ax.text(4.5, 32, 'CS vs. 91F02: ' +str(round_to_significant_digit(result2.pvalue,2)), color='pink')
    ax.text(4.5, 30, 'CS vs. 74C10: ' +str(round_to_significant_digit(result3.pvalue,2)), color='pink')

    yMax = 50
    sft = 3
    for ii in range(np.shape(pVals)[0]):
        yy = yMax-ii*sft
        ax.plot([1, ii+2],[yy,yy], color=const.axisColor, linewidth=1)
        if pVals[ii] < 0.001:
            mkr = '***'
        elif pVals[ii] < 0.01:
            mkr = '**'
        elif pVals[ii] < 0.05:
            mkr = '*'
        else: mkr = 'ns'
        ax.text(ii+1.5,yy+sft/10,mkr,color=const.axisColor,fontsize=const.fontSize_axis+1)

    # configure the axes and plot color, etc.
    ax.set_facecolor(const.figColor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(const.axisColor)
    ax.spines['bottom'].set_color(const.axisColor)
    ax.tick_params(direction='in', length=5, width=0.5)
    ax.tick_params(axis='y',colors=const.axisColor)
    ax.set_ylabel('average peak difference',color=const.axisColor,fontsize=const.fontSize_axis)
    # configure the y-axis
    ax.set_ylim([-10,yMax])
    ax.set_yticks([0, 10,20,30,40])
    ax.spines['left'].set_bounds(0, 40) #do not extend y-axis line beyond ticks
    ## configure the x-axis
    ax.set_xlim([0.5,7.5])
    ax.set_xticks([1,2,3,4])
    ax.tick_params(axis='x',colors=const.axisColor,rotation=30)
    ax.spines['bottom'].set_bounds(1, 4) #do not extend x-axis line beyond ticks
    ax.set_xticklabels(xlabels,rotation=30)

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    # save in folder with today's date
    if savefig:
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        fig.savefig(figPath+'/gainTuningActivation_'+cameraView+'.png', facecolor=fig.get_facecolor())
        fig.savefig(figPath+'/gainTuningActivation_'+cameraView+'.pdf', facecolor=fig.get_facecolor())

# plot different between +45 and -45 for R-L tuning curves across inactivation genotypes
# Plots this across all velocities
def plot_gain_tuning_quant(cameraView='frontal',savefig=0,importAnew=0):

    for ii in [0,1,2,3]:
        _, diff_CS = get_gain_tuning_quant(expt='CS_activate',cameraView=cameraView, velToQuant=ii, savefig=0,importAnew=importAnew)
        _, diff_91F02 = get_gain_tuning_quant(expt='91F02_activate', cameraView=cameraView, velToQuant=ii, savefig=0)
        _, diff_18D07 = get_gain_tuning_quant(expt='18D07_activate', cameraView=cameraView, velToQuant=ii, savefig=0)
        _, diff_74C10 = get_gain_tuning_quant(expt='74C10_activate', cameraView=cameraView, velToQuant=ii, savefig=0)
        if ii == 0:
            diff45_CS = np.zeros([np.shape(diff_CS)[0],4]) #MxN matrix, where M=#flies, N=#speeds (0,50,100,200 cm/s)
            diff45_91F02 = np.zeros([np.shape(diff_91F02)[0],4]) #MxN matrix, where M=#flies, N=#speeds (0,50,100,200 cm/s)
            diff45_18D07 = np.zeros([np.shape(diff_18D07)[0],4]) #MxN matrix, where M=#flies, N=#speeds (0,50,100,200 cm/s)
            diff45_74C10 = np.zeros([np.shape(diff_74C10)[0],4]) #MxN matrix, where M=#flies, N=#speeds (0,50,100,200 cm/s)
        diff45_CS[:,ii] = diff_CS
        diff45_91F02[:,ii] = diff_91F02
        diff45_18D07[:,ii] = diff_18D07
        diff45_74C10[:,ii] = diff_74C10

    #one outlier due to tracking errors omitted (not caught by usual trackign error detection):
    diff45_CS[diff45_CS>70] = np.nan

    data = [diff45_CS, diff45_18D07, diff45_91F02, diff45_74C10]
    # Plot the data separately for each speed, with genotypes separated
    fig, ax = plt.subplots(1,4,facecolor=const.figColor,figsize=(10,6))
    plt.suptitle('gain (-45° - +45°)',
     color=const.axisColor,fontsize=const.fontSize_angPair)
    #plot all the data (points w/ avg)
    jitterSft = 0.25
    for ii in [0,1,2,3]:
        ones = np.ones(np.shape(diff45_CS[:,ii])[0])
        jitter = jitterSft*np.random.rand(np.shape(diff45_CS[:,ii])[0])-jitterSft/2
        ax[ii].plot(1*ones+jitter, diff45_CS[:,ii],marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv)#,alpha=const.transparencyAntTrace)
        ax[ii].plot(1, np.nanmean(diff45_CS[:,ii]),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

        ones = np.ones(np.shape(diff45_18D07[:,ii])[0])
        jitter = jitterSft*np.random.rand(np.shape(diff45_18D07[:,ii])[0])-jitterSft/2
        ax[ii].plot(2*ones+jitter, diff45_18D07[:,ii],marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv)#),alpha=const.transparencyAntTrace)
        ax[ii].plot(2, np.nanmean(diff45_18D07[:,ii]),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

        ones = np.ones(np.shape(diff45_91F02[:,ii])[0])
        jitter = jitterSft*np.random.rand(np.shape(diff45_91F02[:,ii])[0])-jitterSft/2
        ax[ii].plot(3*ones+jitter, diff45_91F02[:,ii],marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv)#,alpha=const.transparencyAntTrace)
        ax[ii].plot(3, np.nanmean(diff45_91F02[:,ii]),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

        ones = np.ones(np.shape(diff45_74C10[:,ii])[0])
        jitter = jitterSft*np.random.rand(np.shape(diff45_74C10[:,ii])[0])-jitterSft/2
        ax[ii].plot(4*ones+jitter, diff45_74C10[:,ii],marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv)#,alpha=const.transparencyAntTrace)
        ax[ii].plot(4, np.nanmean(diff45_74C10[:,ii]),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    # Plot the data separately for each speed, with genotypes separated
    fig2, ax2 = plt.subplots(1,facecolor=const.figColor,figsize=(8,6))
    plt.title('gain (-45° - +45°)',
     color=const.axisColor,fontsize=const.fontSize_angPair)
    #plot all the data (points w/ avg)
    jitterSft = 0.25
    for ii in [0,1,2,3]:
        ones = np.ones(np.shape(diff45_CS[:,ii])[0])
        jitter = jitterSft*np.random.rand(np.shape(diff45_CS[:,ii])[0])-jitterSft/2
        ax2.plot((ii+1)*ones+jitter, diff45_CS[:,ii],marker='.',linestyle='None',color=const.markerColor_CS, markersize=const.markerTuningIndv)#,alpha=const.transparencyAntTrace)

        ones = np.ones(np.shape(diff45_18D07[:,ii])[0])
        jitter = jitterSft*np.random.rand(np.shape(diff45_18D07[:,ii])[0])-jitterSft/2
        ax2.plot((ii+1)*ones+jitter, diff45_18D07[:,ii],marker='.',linestyle='None',color=const.markerColor_18D07, markersize=const.markerTuningIndv)#),alpha=const.transparencyAntTrace)

        ones = np.ones(np.shape(diff45_91F02[:,ii])[0])
        jitter = jitterSft*np.random.rand(np.shape(diff45_91F02[:,ii])[0])-jitterSft/2
        ax2.plot((ii+1)*ones+jitter, diff45_91F02[:,ii],marker='.',linestyle='None',color=const.markerColor_91F02, markersize=const.markerTuningIndv)#,alpha=const.transparencyAntTrace)

        ones = np.ones(np.shape(diff45_74C10[:,ii])[0])
        jitter = jitterSft*np.random.rand(np.shape(diff45_74C10[:,ii])[0])-jitterSft/2
        ax2.plot((ii+1)*ones+jitter, diff45_74C10[:,ii],marker='.',linestyle='None',color=const.markerColor_74C10, markersize=const.markerTuningIndv)#,alpha=const.transparencyAntTrace)
        #ax2.plot(4, np.nanmean(diff45_74C10[:,ii]),marker='.',linestyle='None',color=const.markerColor_74C10, markersize=const.markerTuningAvg+1)

    meanCS = np.nanmean(diff45_CS[:,:],axis=0)
    mean18D07 = np.nanmean(diff45_18D07[:,:],axis=0)
    mean91F02 = np.nanmean(diff45_91F02[:,:],axis=0)
    mean74C10 = np.nanmean(diff45_74C10[:,:],axis=0)
    ax2.plot([1,2,3,4], meanCS,marker='.',linestyle='-',color=const.markerColor_CS, markersize=const.markerTuningAvg+1)
    ax2.plot([1,2,3,4], mean18D07,marker='.',linestyle='-',color=const.markerColor_18D07, markersize=const.markerTuningAvg+1)
    ax2.plot([1,2,3,4], mean91F02,marker='.',linestyle='-',color=const.markerColor_91F02, markersize=const.markerTuningAvg+1)
    ax2.plot([1,2,3,4], mean74C10,marker='.',linestyle='-',color=const.markerColor_74C10, markersize=const.markerTuningAvg+1)
    # print legend for colors/genotypes
    ax2.text(0.9, 32, '18D07>Chrimson', color=const.markerColor_18D07)
    ax2.text(0.9, 34, 'Canton-S>Chrimson', color= const.markerColor_CS)
    ax2.text(0.9, 30, '91F02>Chrimson', color=const.markerColor_91F02)
    ax2.text(0.9, 28, '74C10>Chrimson', color=const.markerColor_74C10)

    # for every speed ind, compare differences vs. control, and configure axes:
    yMax = 50
    sft = 3
    xlabels = ['control','MN line 1', 'MN line 2', 'muscle line']
    xlabels2 = ['0 cm/s','50 cm/s', '100 cm/s', '200 cm/s']
    for ii in [0,1,2,3]:
        dCS = diff45_CS[:,ii]
        dCS = dCS[~np.isnan(dCS)]
        d18D07 = diff45_18D07[:,ii]
        d18D07 = d18D07[~np.isnan(d18D07)]
        d91F02 = diff45_91F02[:,ii]
        d91F02 = d91F02[~np.isnan(d91F02)]
        d74C10 = diff45_74C10[:,ii]
        d74C10 = d74C10[~np.isnan(d74C10)]
        result1 = stats.ttest_ind(dCS, d18D07)
        result2 = stats.ttest_ind(dCS, d91F02)
        result3 = stats.ttest_ind(dCS, d74C10)
        pVals = [result1.pvalue, result2.pvalue, result3.pvalue]
        print(const.speedLabels[ii] + ' pVals: ' + str(pVals))
        ax[ii].text(3.5, 34, 'CSv18: ' +str(round_to_significant_digit(result1.pvalue,3)), color='pink')
        ax[ii].text(3.5, 32, 'CSv91: ' +str(round_to_significant_digit(result2.pvalue,3)), color='pink')
        ax[ii].text(3.5, 30, 'CSv74: ' +str(round_to_significant_digit(result3.pvalue,3)), color='pink')
        # plot result of ttest between groups (indicate significane level)
        for jj in range(np.shape(pVals)[0]):
            yy = yMax-jj*sft
            ax[ii].plot([1, jj+2],[yy,yy], color=const.axisColor, linewidth=1)
            if pVals[jj] < 0.001:
                mkr = '***'
            elif pVals[jj] < 0.01:
                mkr = '**'
            elif pVals[jj] < 0.05:
                mkr = '*'
            else: mkr = 'ns'
            ax[ii].text(ii+1.5,yy+sft/10,mkr,color=const.axisColor,fontsize=const.fontSize_axis+1)

        # configure the axes and plot color, etc.
        ax[ii].set_facecolor(const.figColor)
        ax[ii].spines['right'].set_visible(False)
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['left'].set_color(const.axisColor)
        ax[ii].spines['bottom'].set_color(const.axisColor)
        ax[ii].tick_params(direction='in', length=5, width=0.5)
        ax[ii].tick_params(axis='y',colors=const.axisColor)
        ax[ii].set_ylabel('average peak difference',color=const.axisColor,fontsize=const.fontSize_axis)
        # configure the y-axis
        ax[ii].set_ylim([-10,yMax])
        ax[ii].set_yticks([0, 10,20,30,40])
        ax[ii].spines['left'].set_bounds(0, 40) #do not extend y-axis line beyond ticks
        ## configure the x-axis
        ax[ii].set_xlim([0.5,7.5])
        ax[ii].set_xticks([1,2,3,4])
        ax[ii].tick_params(axis='x',colors=const.axisColor,rotation=30)
        ax[ii].spines['bottom'].set_bounds(1, 4) #do not extend x-axis line beyond ticks
        ax[ii].set_xticklabels(xlabels,rotation=30)

        # configure the axes and plot color, etc.
        ax2.set_facecolor(const.figColor)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_color(const.axisColor)
        ax2.spines['bottom'].set_color(const.axisColor)
        ax2.tick_params(direction='in', length=5, width=0.5)
        ax2.tick_params(axis='y',colors=const.axisColor)
        ax2.set_ylabel('average peak difference',color=const.axisColor,fontsize=const.fontSize_axis)
        # configure the y-axis
        ax2.set_ylim([-10,yMax])
        ax2.set_yticks([0, 10,20,30,40])
        ax2.spines['left'].set_bounds(0, 40) #do not extend y-axis line beyond ticks
        ## configure the x-axis
        ax2.set_xlim([0.5,7.5])
        ax2.set_xticks([1,2,3,4])
        ax2.tick_params(axis='x',colors=const.axisColor,rotation=30)
        ax2.spines['bottom'].set_bounds(1, 4) #do not extend x-axis line beyond ticks
        ax2.set_xticklabels(xlabels2,rotation=0)


    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    # save in folder with today's date
    if savefig:
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        fig.savefig(figPath+'/gainTuningActivationAllVelSeparated_'+cameraView+'.png', facecolor=fig.get_facecolor())
        fig.savefig(figPath+'/gainTuningActivationAllVelSeparated_'+cameraView+'.pdf', facecolor=fig.get_facecolor())

        fig2.savefig(figPath+'/gainTuningActivationAllVel_'+cameraView+'.png', facecolor=fig2.get_facecolor())
        fig2.savefig(figPath+'/gainTuningActivationAllVel_'+cameraView+'.pdf', facecolor=fig2.get_facecolor())


# plot different between +45 and -45 for R-L tuning curves - incativation
def plot_gain_tuning_quant_inactivation(cameraView='frontal',savefig=0):
    indNeg45 = 1
    indPos45 = 3
    velToQuant = 1

    expt = 'emptyGAL4_inactivate'
    avgTuningRAllTrials = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningRAllTrials.npy')
    avgTuningLAllTrials = np.load( const.savedDataDirectory+expt+'_'+'avgDirectionTuningLAllTrials.npy')
    avgTuningRAllTrials_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningRAllTrials_lightOff.npy')
    avgTuningLAllTrials_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningLAllTrials_lightOff.npy')
    avgTuningR = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningR.npy')
    avgTuningL = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningL.npy')
    avgTuningR_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningR_lightOff.npy')
    avgTuningL_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningL_lightOff.npy')
    RminL_empty_nolight = avgTuningR_lightOff[velToQuant][:]-avgTuningL_lightOff[velToQuant][:]
    RminL_200cm_lightOff = avgTuningRAllTrials_lightOff[:,velToQuant,:]-avgTuningLAllTrials_lightOff[:,velToQuant,:]
    diff45_empty_nolight = RminL_200cm_lightOff[:,indNeg45]-RminL_200cm_lightOff[:,indPos45]
    RminL_empty_light = avgTuningR[velToQuant][:]-avgTuningL[velToQuant][:]
    RminL_200cm = avgTuningRAllTrials[:,velToQuant,:]-avgTuningLAllTrials[:,velToQuant,:]
    diff45_empty_light = RminL_200cm[:,indNeg45]-RminL_200cm[:,indPos45]

    expt = '18D07_inactivate'
    avgTuningRAllTrials = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningRAllTrials.npy')
    avgTuningLAllTrials = np.load( const.savedDataDirectory+expt+'_'+'avgDirectionTuningLAllTrials.npy')
    avgTuningRAllTrials_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningRAllTrials_lightOff.npy')
    avgTuningLAllTrials_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningLAllTrials_lightOff.npy')
    avgTuningR = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningR.npy')
    avgTuningL = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningL.npy')
    avgTuningR_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningR_lightOff.npy')
    avgTuningL_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningL_lightOff.npy')
    RminL_18D07_nolight = avgTuningR_lightOff[velToQuant][:]-avgTuningL_lightOff[velToQuant][:]
    RminL_200cm_lightOff = avgTuningRAllTrials_lightOff[:,velToQuant,:]-avgTuningLAllTrials_lightOff[:,velToQuant,:]
    diff45_18D07_nolight = RminL_200cm_lightOff[:,indNeg45]-RminL_200cm_lightOff[:,indPos45]
    RminL_18D07_light = avgTuningR[velToQuant][:]-avgTuningL[velToQuant][:]
    RminL_200cm = avgTuningRAllTrials[:,velToQuant,:]-avgTuningLAllTrials[:,velToQuant,:]
    diff45_18D07_light = RminL_200cm[:,indNeg45]-RminL_200cm[:,indPos45]

    expt = '91F02_inactivate'
    avgTuningRAllTrials = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningRAllTrials.npy')
    avgTuningLAllTrials = np.load( const.savedDataDirectory+expt+'_'+'avgDirectionTuningLAllTrials.npy')
    avgTuningRAllTrials_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningRAllTrials_lightOff.npy')
    avgTuningLAllTrials_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningLAllTrials_lightOff.npy')
    avgTuningR = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningR.npy')
    avgTuningL = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningL.npy')
    avgTuningR_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningR_lightOff.npy')
    avgTuningL_lightOff = np.load(const.savedDataDirectory+expt+'_'+'avgDirectionTuningL_lightOff.npy')
    RminL_91F02_nolight = avgTuningR_lightOff[velToQuant][:]-avgTuningL_lightOff[velToQuant][:]
    RminL_200cm_lightOff = avgTuningRAllTrials_lightOff[:,velToQuant,:]-avgTuningLAllTrials_lightOff[:,velToQuant,:]
    diff45_91F02_nolight = RminL_200cm_lightOff[:,indNeg45]-RminL_200cm_lightOff[:,indPos45]

    RminL_91F02_light = avgTuningR[velToQuant][:]-avgTuningL[velToQuant][:]
    RminL_200cm = avgTuningRAllTrials[:,velToQuant,:]-avgTuningLAllTrials[:,velToQuant,:]
    diff45_91F02_light = RminL_200cm[:,indNeg45]-RminL_200cm[:,indPos45]

    #difference between no light and light
    difference_emptyGAL4 = diff45_empty_nolight-diff45_empty_light
    difference_91F02 = diff45_91F02_nolight-diff45_91F02_light
    difference_18D07 = diff45_18D07_nolight-diff45_18D07_light

    diff45_empty_light = diff45_empty_light[~np.isnan(diff45_empty_light)]
    diff45_91F02_light = diff45_91F02_light[~np.isnan(diff45_91F02_light)]
    diff45_18D07_light = diff45_18D07_light[~np.isnan(diff45_18D07_light)]

    #grab data with light off
    diff45_empty_nolight = diff45_empty_nolight[~np.isnan(diff45_empty_nolight)]
    diff45_91F02_nolight = diff45_91F02_nolight[~np.isnan(diff45_91F02_nolight)]
    diff45_18D07_nolight = diff45_18D07_nolight[~np.isnan(diff45_18D07_nolight)]

    xlabels = ['control\nnolight','18D07\nnolight', '91F02\nnolight',
    'control\n+light','18D07\n+light', '91F02\n+light',
    'control\ndiff','18D07\ndiff', '91F02\ndiff']
    # Plot the data
    fig, ax = plt.subplots(1,1,facecolor=const.figColor,figsize=(8,10))
    plt.suptitle('gain (-45° - +45°)',
     color=const.axisColor,fontsize=const.fontSize_angPair)

    #plot all the with light data (points w/ avg)
    jitterSft = 0.25
    ones = np.ones(np.shape(diff45_empty_nolight)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_empty_nolight)[0])-jitterSft/2
    ax.plot(1*ones+jitter, diff45_empty_nolight,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(1, np.nanmean(diff45_empty_nolight),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(diff45_18D07_nolight)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_18D07_nolight)[0])-jitterSft/2
    ax.plot(2*ones+jitter, diff45_18D07_nolight,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(2, np.nanmean(diff45_18D07_nolight),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(diff45_91F02_nolight)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_91F02_nolight)[0])-jitterSft/2
    ax.plot(3*ones+jitter, diff45_91F02_nolight,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(3, np.nanmean(diff45_91F02_nolight),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    #plot all the with light data (points w/ avg)
    jitterSft = 0.25
    ones = np.ones(np.shape(diff45_empty_light)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_empty_light)[0])-jitterSft/2
    ax.plot(5*ones+jitter, diff45_empty_light,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(5, np.nanmean(diff45_empty_light),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(diff45_18D07_light)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_18D07_light)[0])-jitterSft/2
    ax.plot(6*ones+jitter, diff45_18D07_light,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(6, np.nanmean(diff45_18D07_light),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(diff45_91F02_light)[0])
    jitter = jitterSft*np.random.rand(np.shape(diff45_91F02_light)[0])-jitterSft/2
    ax.plot(7*ones+jitter, diff45_91F02_light,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(7, np.nanmean(diff45_91F02_light),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    #plot all difference between light and no light (lose a few flies, some had nan due to flight during trials!)
    jitterSft = 0.25
    ones = np.ones(np.shape(difference_emptyGAL4)[0])
    jitter = jitterSft*np.random.rand(np.shape(difference_emptyGAL4)[0])-jitterSft/2
    ax.plot(9*ones+jitter, difference_emptyGAL4,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(9, np.nanmean(difference_emptyGAL4),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(difference_18D07)[0])
    jitter = jitterSft*np.random.rand(np.shape(difference_18D07)[0])-jitterSft/2
    ax.plot(10*ones+jitter, difference_18D07,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(10, np.nanmean(difference_18D07),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    ones = np.ones(np.shape(difference_91F02)[0])
    jitter = jitterSft*np.random.rand(np.shape(difference_91F02)[0])-jitterSft/2
    ax.plot(11*ones+jitter, difference_91F02,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
    ax.plot(11, np.nanmean(difference_91F02),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    # plot result of ttest between groups (indicate significane level)
    print(difference_emptyGAL4)
    print(difference_18D07)
    print(difference_91F02)
    result1 = stats.ttest_ind(difference_emptyGAL4[~np.isnan(difference_emptyGAL4)], difference_18D07[~np.isnan(difference_18D07)])
    result2 = stats.ttest_ind(difference_emptyGAL4[~np.isnan(difference_emptyGAL4)],  difference_91F02[~np.isnan(difference_91F02)])
    pVals = [result1.pvalue, result2.pvalue]
    ax.text(4.5, 54, 'empty vs. 91F02: ' +str(round_to_significant_digit(result1.pvalue,2)), color='pink')
    ax.text(4.5, 52, 'empty vs. 18D07: ' +str(round_to_significant_digit(result2.pvalue,2)), color='pink')

    yMax = 70
    sft = 3
    for ii in range(np.shape(pVals)[0]):
        yy = yMax-ii*sft
        ax.plot([9, 9+ii+1],[yy,yy], color=const.axisColor, linewidth=1)
        if pVals[ii] < 0.001:
            mkr = '***'
        elif pVals[ii] < 0.01:
            mkr = '**'
        elif pVals[ii] < 0.05:
            mkr = '*'
        else: mkr = 'ns'
        ax.text(ii+9.5,yy+sft/10,mkr,color=const.axisColor,fontsize=const.fontSize_axis+1)

    # configure the axes and plot color, etc.
    ax.set_facecolor(const.figColor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(const.axisColor)
    ax.spines['bottom'].set_color(const.axisColor)
    ax.tick_params(direction='in', length=5, width=0.5)
    ax.tick_params(axis='y',colors=const.axisColor)
    ax.set_ylabel('average peak difference',color=const.axisColor,fontsize=const.fontSize_axis)
    # configure the y-axis
    ax.set_ylim([-10,yMax])
    ax.set_yticks([0, 10,20,30,40,50,60])
    ax.spines['left'].set_bounds(0, 60) #do not extend y-axis line beyond ticks
    ## configure the x-axis
    #ax.set_xlim([0.5,7.5])
    ax.set_xlim([0.5,11.5])
    ax.set_xticks([1,2,3,5,6,7,9,10,11])
    ax.tick_params(axis='x',colors=const.axisColor,rotation=30)
    ax.spines['bottom'].set_bounds(1, 11) #do not extend x-axis line beyond ticks
    ax.set_xticklabels(xlabels,rotation=30)

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    extraText = ''

    # save in folder with today's date
    if savefig:
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        fig.savefig(figPath+'/gainTuningActivation_'+cameraView+extraText+'.png', facecolor=fig.get_facecolor())
        fig.savefig(figPath+'/gainTuningActivation_'+cameraView+extraText+'.pdf', facecolor=fig.get_facecolor())

# return single trial and average antenna angles at light onset for one fly
# antenna angles are baseline-subtracted
def get_single_expt_lightResponse(expt='2020_12_14_E3', cameraView='frontal',importAnew=0,savefig=0,allowFlight=0,plotInactivate=0):
    exptNotes = get_mat_notes(expt)
    windVel = exptNotes.velocity[:]
    windVel = windVel.str.get(0)  # take values out of brackets
    [angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight] = getAntennaTracesAvgs_singleExpt(expt, cameraView, importAnew,allowFlight,plotInactivate=plotInactivate)
    angs_all = detect_tracking_errors_raw_trace(exptNotes, expt, angs_all)
    framerate = const.framerate
    # stimStart = const.activateAvgSt
    preLightInd = int(const.preLightOn*framerate)
    postLightInd = int(const.postLightOn*framerate)
    preLightOffInd = int(const.preLightOff*framerate)
    postLightOffInd = int(const.postLightOff*framerate)
    rangeTraceOn = range(preLightInd, postLightInd-3)  # how much of trace to plot (start to onset of wind)
    rangeTraceOff = range(preLightOffInd, postLightOffInd)  # how much of trace to plot (at light off)
    rangeFullTrace = range(const.lenVideo*const.framerate)

    numTrials = angs_all.shape[0]
    angsOn_singleTrials = np.empty([numTrials, np.shape(rangeTraceOn)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    angsOn_avg = np.empty([np.shape(rangeTraceOn)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    angsOff_singleTrials = np.empty([numTrials, np.shape(rangeTraceOff)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    angsOff_avg = np.empty([np.shape(rangeTraceOff)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    base = np.empty([np.shape(const.angPairNames[cameraView])[0]])
    tracesAllIndv = np.empty([numTrials, np.shape(rangeFullTrace)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    tracesAvg = np.empty([np.shape(rangeFullTrace)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    #baseWholeTrace = np.empty([np.shape(const.angPairNames[cameraView])[0]])
    baseWholeTrace_allTrials = np.empty([numTrials,np.shape(const.angPairNames[cameraView])[0]])

    for ang in range(angs_all.shape[2]):
        for exptNum in range(angs_all.shape[0]):
            baseline = avgBase[exptNum][ang]
            angsOn_singleTrials[exptNum,:,ang]= angs_all[exptNum, rangeTraceOn, ang]-baseline
            angsOff_singleTrials[exptNum,:,ang]= angs_all[exptNum, rangeTraceOff, ang]-baseline
            # only include full traces for 0-velocity trials (activation-only, no wind)
            if(windVel[exptNum]==0):
                tracesAllIndv[exptNum,:,ang] = angs_all[exptNum, rangeFullTrace, ang]-baseline
                baseWholeTrace_allTrials[exptNum,ang] = avgBase[exptNum,ang]
            else:
                emptyTrace = np.empty(np.shape(angs_all[exptNum, rangeFullTrace, ang]))
                emptyTrace[:] = np.nan
                tracesAllIndv[exptNum,:,ang] = emptyTrace
                baseWholeTrace_allTrials[exptNum,ang] = np.nan
        base[ang] = np.nanmean(avgBase[:,ang])
        angsOn_avg[:,ang] = np.nanmean(angs_all[:, rangeTraceOn, ang],0)
        angsOff_avg[:,ang] = np.nanmean(angs_all[:, rangeTraceOff, ang],0)
        tracesAvg[:,ang] = np.nanmean(tracesAllIndv[:, :, ang],0)

    return angsOn_avg, angsOn_singleTrials, angsOff_avg, angsOff_singleTrials, base, tracesAvg, tracesAllIndv, baseWholeTrace_allTrials

# Acquire cross-fly averages for direction and velocity tuning
#  for a given genotype/experiment set
def get_cross_expt_lightResponse(expt='74C10_activate', cameraView='frontal',importAnew=0,allowFlight=0,plotInactivate=0):
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)
    framerate = const.framerate
    stimStart = const.activateAvgSt
    preLightInd = int(const.preLightOn*framerate)
    postLightInd = int(const.postLightOn*framerate)
    rangeTraceOn = range(preLightInd, postLightInd-3) #how much of trace to plot (start to onset of wind)
    preLightOffInd = int(const.preLightOff*framerate)
    postLightOffInd = int(const.postLightOff*framerate)
    rangeTraceOff = range(preLightOffInd, postLightOffInd) #how much of trace to plot (at light off)
    rangeFullTrace = range(const.lenVideo*const.framerate)

    # NxMxP array, where N = # flies (variable, ~10)
    tracesAllIndv = np.empty([np.shape(allExpts)[0],np.shape(rangeFullTrace)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    onsetAllFlies = np.empty([np.shape(allExpts)[0],np.shape(rangeTraceOn)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    offsetAllFlies = np.empty([np.shape(allExpts)[0],np.shape(rangeTraceOff)[0],
        np.shape(const.angPairNames[cameraView])[0]])
    baseAllFlies = np.empty([np.shape(allExpts)[0],np.shape(const.angPairNames[cameraView])[0]])
    baseAllFliesFullTrace = np.empty([np.shape(allExpts)[0],np.shape(const.angPairNames[cameraView])[0]])
    # Acquire list of experiments for this genotype
    for noteInd, notes in enumerate(allExpts):
        flyExpt = notes['date']
        # get tuning responses for single fly
        [angsOn_avg, angsOn_singleTrials, angsOff_avg, angsOff_singleTrials, base, avgFullTrace, angsFullTrace,baseWholeTrace] = get_single_expt_lightResponse(flyExpt, cameraView, importAnew=importAnew, savefig=0,allowFlight=allowFlight,plotInactivate=plotInactivate)
        tracesAllIndv[noteInd,:]= avgFullTrace
        onsetAllFlies[noteInd,:] = angsOn_avg
        offsetAllFlies[noteInd,:] = angsOff_avg
        baseAllFlies[noteInd,:] = base
        baseAllFliesFullTrace[noteInd,:] = np.nanmean(baseWholeTrace,0)
    onsetAvg = np.nanmean(onsetAllFlies,0)
    offsetAvg = np.nanmean(offsetAllFlies,0)
    tracesAvg= np.nanmean(tracesAllIndv,0)

    return onsetAvg, onsetAllFlies, offsetAvg, offsetAllFlies, baseAllFlies, tracesAvg, tracesAllIndv, baseAllFliesFullTrace

# Plot average traces at light onset and offset for a single fly or across a genotype
# Something strange is going on with baseline values - baseline subtraction appears tachOverThreshPercent
#  for some traces, but does not look right for others? Also some irregularities with traces? What's happening?
#  In contrast, everythign looks correct with entire trace in plot_lightResponse
def plot_lightOnsetOffset(expt='2020_12_14_E3', cameraView='frontal',plotInactivate=0,savefig=0):
    import loadNotes_mnOptoVelocityTuning as ln

    try: # if 'expt' input is a genotype/experiment type, then will plot cross-fly
        allExpts = eval('ln.notes_'+expt)
        plot_single_fly = 0
        [onsetAvg, onsetAllFlies, offsetAvg, offsetAllFlies, baseVals, tracesAvg, tracesAllIndv, baseAllFliesFullTrace] = get_cross_expt_lightResponse(expt, cameraView, 0,plotInactivate=plotInactivate)
    except: # if 'expt' input is a date/fly, plot traces from this single fly
        [angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight] = getAntennaTracesAvgs_singleExpt(expt, cameraView, 0,plotInactivate=plotInactivate)
        [onsetAvg, onsetAllFlies, offsetAvg, offsetAllFlies, baseVals, tracesAvg, tracesAllIndv, baseAllFliesFullTrace] = get_single_expt_lightResponse(expt, cameraView, 0,plotInactivate=plotInactivate)
        baseVals = avgBase
        plot_single_fly = 1

    fig, axAng = plt.subplots(facecolor=const.figColor,figsize=(6,10))
    # define some axis sizing and data regions:
    scaleBarSize = 10
    scaleWidth = 1
    scaleY = 5
    framerate = const.framerate
    stimStart = const.activateAvgSt
    preLightInd = int(const.preLightOn*framerate)
    postLightInd = int(const.postLightOn*framerate)
    rangeTrace = range(preLightInd, postLightInd-3) #how much of trace to plot (start to onset of wind)
    scaleX = -5 #put vertical scale bar to the right of the traces
    #baseInds = range(0,1*const.framerate) #baseline is first second of trace (pre-activation)
    baseInds = range(0,int(const.preLightOn*framerate)) #baseline is first second of trace (pre-activation)

    yaxis_max = 50
    yaxis_min = -17
    xaxis_min = -10
    horizontalShift = int(const.framerate/8)#int(const.framerate/2)
    nanarr = np.empty((horizontalShift+np.shape(rangeTrace)[0]))
    nanarr[:] = np.NaN
    xaxis_max = const.lenOnsetLight*framerate+const.lenOffsetLight*framerate+horizontalShift+framerate*2
    # add vertical scale bar
    axAng.add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
        facecolor = const.axisColor))
    axAng.text(scaleX-scaleWidth*10, scaleY+scaleBarSize/2,
        str(scaleBarSize) + const.degree_sign,color=const.axisColor,
        fontsize=const.fontSize_angPair,horizontalalignment='left',
        verticalalignment='center')

    #draw light (onset) activation stimulus bar
    rectX = int(const.preLightOn*const.framerate-1)
    rectY = yaxis_min#-rectHeight*.05
    rectWid = int(const.lenOnsetLight*const.framerate)
    axAng.add_patch(Rectangle((rectX,rectY),rectWid,const.stimBar_height,facecolor = const.color_activateColor))
    axAng.text(rectX+rectWid/2, rectY-const.stimBar_height*2-const.fontSize_stimBar/const.scaleAng_rawTraces/2,
        str(int(const.lenOnsetLight*1000))+' ms light on',color=const.color_activateColor,
        fontsize=const.fontSize_stimBar,horizontalalignment='center')

    #plot offset activation bar
    rectX = np.shape(rangeTrace)[0]+horizontalShift
    rectWid = int(const.lenOffsetLight*const.framerate)
    axAng.add_patch(Rectangle((rectX,rectY),rectWid,const.stimBar_height,facecolor = const.color_activateColor))
    axAng.text(rectX+rectWid/2, rectY-const.stimBar_height*2-const.fontSize_stimBar/const.scaleAng_rawTraces/2,
        str(int(const.lenOffsetLight*1000))+' ms light off',color=const.color_activateColor,
        fontsize=const.fontSize_stimBar,horizontalalignment='center')

    #plot traces for each antenna angle of interest (usually 4)
    for ang in range(onsetAllFlies.shape[2]):
        for exptNum in range(onsetAllFlies.shape[0]):
            baseline = np.nanmean(onsetAllFlies[exptNum, baseInds, ang])
            shift = ang*const.shiftYTraces  # shifts traces relative to each other (ease of viewing)
            plt.plot(onsetAllFlies[exptNum, :, ang]+shift-baseline,
                color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
            baseline = np.nanmean(offsetAllFlies[exptNum, baseInds, ang])
            offsetData = offsetAllFlies[exptNum, :, ang]+shift-baseline
            offsetData = np.concatenate((nanarr, offsetData)) #shift data to right for viewing alongside onset responses
            plt.plot(offsetData,
                color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
        #plot average trace (cross-trial or cross-fly):
        baseline = np.nanmean(onsetAllFlies[:, baseInds, ang])
        plt.plot(onsetAvg[:,ang]+shift-baseline,color=const.colors_antAngsDark[ang], linewidth=5) #, alpha=const.transparencyAntTrace))
        baseline = np.nanmean(offsetAllFlies[:, baseInds, ang])
        offsetData = offsetAvg[:,ang]+shift-baseline
        offsetData = np.concatenate((nanarr, offsetData)) #shift data to right for viewing alongside onset responses
        plt.plot(offsetData,color=const.colors_antAngsDark[ang], linewidth=5) #, alpha=const.transparencyAntTrace))

        labelX = np.shape(onsetAllFlies[exptNum, :, ang])[0]+horizontalShift*2+np.shape(offsetAllFlies[exptNum, :, ang])[0]
        axAng.text(labelX, shift, const.angPairNames[cameraView][ang],
            fontsize=const.fontSize_angPair, color = const.colors_antAngs[ang])

    # Plot title, set axes
    if plot_single_fly:
        fig.suptitle('Light on and offset responses for one fly - '+expt, fontsize=const.fontSize_stimBar)
    else:
        fig.suptitle('Light on and offset responses for all flies - '+expt, fontsize=const.fontSize_stimBar)
    axAng.set_ylim(yaxis_min, yaxis_max)
    axAng.set_xlim(xaxis_min, xaxis_max)
    axAng.axis('off')

    plt.show()
    plt.pause(0.001)
    plt.show(block=False)

    if plotInactivate == 1:
        extraText = '_withLight_'
    elif plotInactivate == 2:
        extraText = '_noLight_'
    else:
        extraText = ''

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath + '_onOffsetTraces_' + expt + extraText + '.png'
        print('Saving figure here: ' + savepath)
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + '_onOffsetTraces_' + expt + extraText + '.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())


# Plot light-induced antennal response for single or cross-flies
def plot_lightResponse(expt='2020_12_14_E3', cameraView='frontal',plotInactivate=0,savefig=0,importAnew=0):
    import SuverEtAl2023_loadNotes as ln
    print(expt)
    try: # if 'expt' input is a genotype/experiment type, then will plot cross-fly
        allExpts = eval('ln.notes_'+expt)
        plot_single_fly = 0
        [onsetAvg, onsetAllFlies, offsetAvg, offsetAllFlies, baseVals,
            tracesAvg, tracesAllIndv, baseWholeTrace] = get_cross_expt_lightResponse(expt, cameraView,importAnew=importAnew,plotInactivate=plotInactivate)
    except: # if 'expt' input is a date/fly, plot traces from this single fly
        [angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight] = getAntennaTracesAvgs_singleExpt(expt, cameraView, 0,plotInactivate=plotInactivate)
        [angsOn_avg, onsetAllFlies, angsOff_avg, offsetAllFlies, baseVals,
            tracesAvg, tracesAllIndv, baseWholeTrace] = get_single_expt_lightResponse(expt, cameraView, 0,plotInactivate=plotInactivate)
        baseVals = baseWholeTrace
        plot_single_fly = 1

    fig, axAng = plt.subplots(facecolor=const.figColor,figsize=(10,8))
    # define some axis sizing and data regions:
    scaleBarSize = 10
    scaleWidth = 1
    scaleY = 5
    framerate = const.framerate
    stimStart = const.activateAvgSt
    preLightInd = int(const.preLightOn*framerate)
    postLightInd = int(const.postLightOn*framerate)
    steadyStateStimInds = list(range(2*const.framerate, 6*const.framerate)) # beginning 1 sec after light to end of light (4 sec light)

    rangeTrace = range(preLightInd, postLightInd-3) #how much of trace to plot (start to onset of wind)
    scaleX = -5 #put vertical scale bar to the right of the traces

    if cameraView == 'frontal':
        yaxis_max = 50
    else:
        yaxis_max = 70
    yaxis_min = -15
    xaxis_min = -10
    xaxis_max = const.lenVideo*const.framerate+const.framerate
    # add vertical scale bar
    axAng.add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
        facecolor = const.axisColor))
    axAng.text(scaleX-scaleWidth*6, scaleY+scaleBarSize/2,
        str(scaleBarSize) + const.degree_sign,color=const.axisColor,
        fontsize=const.fontSize_angPair,horizontalalignment='left',
        verticalalignment='center')

    #draw light activation stimulus bar
    rectX = int(const.activateStart*const.framerate)
    rectY = yaxis_min#-rectHeight*.05
    rectWid = int(const.activateTime*const.framerate)
    axAng.add_patch(Rectangle((rectX,rectY),rectWid,const.stimBar_height,facecolor = const.color_activateColor))
    axAng.text(rectX+rectWid/2, rectY-const.stimBar_height*2-const.fontSize_stimBar/const.scaleAng_rawTraces/2,
        str(int(const.activateTime))+' s light on',color=const.color_activateColor,
        fontsize=const.fontSize_stimBar,horizontalalignment='center')

    #plot traces for each antenna angle of interest (usually 4)
    for ang in range(tracesAllIndv.shape[2]):
        for exptNum in range(tracesAllIndv.shape[0]):
            baseline = baseVals[exptNum][ang]
            shift = ang*const.shiftYTraces  # shifts traces relative to each other (ease of viewing)
            axAng.plot(tracesAllIndv[exptNum, :, ang]+shift,
                color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)

        #plot average trace (cross-trial or cross-fly):
        base = np.nanmean(baseVals[:,ang])
        axAng.plot(tracesAvg[:,ang]+shift,color=const.colors_antAngsDark[ang], linewidth=5) #, alpha=const.transparencyAntTrace))
        axAng.text(const.lenVideo*const.framerate+const.framerate/10, shift, const.angPairNames[cameraView][ang],
            fontsize=const.fontSize_angPair, color = const.colors_antAngs[ang])

    # Plot title, set axes
    if plot_single_fly:
        fig.suptitle('Light responses for one fly - '+expt, fontsize=const.fontSize_stimBar)
    else:
        fig.suptitle('Light responses for all flies - '+expt, fontsize=const.fontSize_stimBar)
    axAng.set_ylim(yaxis_min, yaxis_max)
    axAng.set_xlim(xaxis_min, xaxis_max)
    axAng.axis('off')

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if plotInactivate == 1:
        extraText = '_withLight_'
    elif plotInactivate == 2:
        extraText = '_noLight_'
    else:
        extraText = ''

    # save antenna traces for quantification later (in next function: plot_lightResponse_quant())
    ang_fn = const.savedDataDirectory+expt+extraText+'lightTracesIndv.npy'
    np.save(ang_fn,tracesAllIndv)
    ang_fn = const.savedDataDirectory+expt+extraText+'lightTracesAvg.npy'
    np.save(ang_fn,tracesAvg)

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath + cameraView +'_activationTraces_' + expt + extraText+ '.png'
        print('Saving figure here: ' + savepath)
        fig.savefig(savepath, facecolor=fig.get_facecolor())

        savepath = figPath + cameraView +'_activationTraces_' + expt + extraText+'.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())


# Plot light-induced antennal deflection averages (single flies and cross-flies)
def plot_lightResponseMeanDeflection(expt='18D07_inactivate', cameraView='frontal',plotInactivate=1,savefig=0):

    import loadNotes_mnOptoVelocityTuning as ln
    fig, axAng = plt.subplots(facecolor=const.figColor,figsize=(10,8))
    steadyStateStimInds = list(range(2*const.framerate, 6*const.framerate)) # beginning 1 sec after light to end of light (4 sec light)

    [onsetAvg, onsetAllFlies, offsetAvg, offsetAllFlies, baseVals,
        tracesAvg, tracesAllIndv, baseWholeTrace] = get_cross_expt_lightResponse(expt, cameraView,importAnew=0,plotInactivate=plotInactivate)
    #plot traces for each antenna angles of interest - left and right segments
    for ang in [0,2]:# range(tracesAllIndv.shape[2]):
        for exptNum in range(tracesAllIndv.shape[0]):
            print(np.nanmean(tracesAllIndv[exptNum, steadyStateStimInds, ang]))
            axAng.plot(np.nanmean(tracesAllIndv[exptNum, steadyStateStimInds, ang]),
                marker='.',linestyle='None', markersize=const.markerTuningIndv,
                color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)

    # Plot title, set axes
    fig.suptitle('Average antennal deflection for ' + expt, fontsize=const.fontSize_stimBar)

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if plotInactivate == 1:
        extraText = '_withLight_'
    elif plotInactivate == 2:
        extraText = '_noLight_'
    else:
        extraText = ''

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath + cameraView +'_activationTraces_' + expt + extraText+ '.png'
        print('Saving figure here: ' + savepath)
        fig.savefig(savepath, facecolor=fig.get_facecolor())

        savepath = figPath + cameraView +'_activationTraces_' + expt + extraText+'.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())

# Plot light induced antennal deflection averages for inactivation experiments, second segment only
#  and compare spread of distributions
def plot_light_response_quant_inactivation(savefig=0):

    fig, ax = plt.subplots(1,1,facecolor=const.figColor,figsize=(4,6))

    extraText = '_withLight_' #no light is just a flat line, no need to quantify here
    expt1 = 'emptyGAL4_inactivate'
    ang_fn = const.savedDataDirectory+expt1+extraText+'lightTracesIndv.npy'
    tracesAllIndv_emptyGAL4_inactivate = np.load(ang_fn)
    expt2 = '18D07_inactivate'
    ang_fn = const.savedDataDirectory+expt2+extraText+'lightTracesIndv.npy'
    tracesAllIndv_18D07_inactivate = np.load(ang_fn)
    expt3 = '91F02_inactivate'
    ang_fn = const.savedDataDirectory+expt3+extraText+'lightTracesIndv.npy'
    tracesAllIndv_91F02_inactivate = np.load(ang_fn)

    avgIndsStart = const.framerate*(const.activateStart+3)
    avgIndsStop = const.framerate*(const.activateStart+5)
    avgInds = range(avgIndsStart, avgIndsStop,1)  #steady-state, last 2 seconds of 5 s light pulse
    baseInds = range(1,const.framerate*1-1,1) #baseline pre-light range (will compute deviation from this)

    for ang in [0,2]: #only plot for 2nd segments
        #plot emptyGAL4_inactivate
        flyAvgTraces = tracesAllIndv_emptyGAL4_inactivate[:, avgInds, ang]
        flyBaseTraces = tracesAllIndv_emptyGAL4_inactivate[:, baseInds, ang]
        flyAvgDeflections = np.nanmean(flyBaseTraces,1)-np.nanmean(flyAvgTraces,1)

        jitterSft = 0.25
        ones = np.ones(np.shape(flyAvgDeflections)[0])
        jitter = jitterSft*np.random.rand(np.shape(flyAvgDeflections)[0])-jitterSft/2
        ax.plot(1*ones+jitter, flyAvgDeflections,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)

        #plot 18D07_inactivate
        flyAvgTraces = tracesAllIndv_18D07_inactivate[:, avgInds, ang]
        flyBaseTraces = tracesAllIndv_18D07_inactivate[:, baseInds, ang]
        flyAvgDeflections = np.nanmean(flyBaseTraces,1)-np.nanmean(flyAvgTraces,1)
        jitterSft = 0.25
        ones = np.ones(np.shape(flyAvgDeflections)[0])
        jitter = jitterSft*np.random.rand(np.shape(flyAvgDeflections)[0])-jitterSft/2
        ax.plot(3*ones+jitter, flyAvgDeflections,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)

        #plot 91F02_inactivate
        flyAvgTraces = tracesAllIndv_91F02_inactivate[:, avgInds, ang]
        flyBaseTraces = tracesAllIndv_91F02_inactivate[:, baseInds, ang]
        flyAvgDeflections = np.nanmean(flyBaseTraces,1)-np.nanmean(flyAvgTraces,1)
        print(flyAvgDeflections)
        jitterSft = 0.25
        ones = np.ones(np.shape(flyAvgDeflections)[0])
        jitter = jitterSft*np.random.rand(np.shape(flyAvgDeflections)[0])-jitterSft/2
        ax.plot(5*ones+jitter, flyAvgDeflections,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)

    allEmpty = np.concatenate((np.nanmean(tracesAllIndv_emptyGAL4_inactivate[:, baseInds, 0],1)-np.nanmean(tracesAllIndv_emptyGAL4_inactivate[:, avgInds, 0],1),
                np.nanmean(tracesAllIndv_emptyGAL4_inactivate[:, baseInds, 2],1)-np.nanmean(tracesAllIndv_emptyGAL4_inactivate[:, avgInds, 2],1)))
    avgEmpty = np.nanmean(allEmpty)
    ax.plot(1, np.nanmean(avgEmpty),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    all18D07 = np.concatenate((np.nanmean(tracesAllIndv_18D07_inactivate[:, baseInds, 0],1)-np.nanmean(tracesAllIndv_18D07_inactivate[:, avgInds, 0],1),
                np.nanmean(tracesAllIndv_18D07_inactivate[:, baseInds, 2])-np.nanmean(tracesAllIndv_18D07_inactivate[:, avgInds, 2],1)))
    avg18D07 = np.nanmean(all18D07)
    ax.plot(3, np.nanmean(avg18D07),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    all91F02 = np.concatenate((np.nanmean(tracesAllIndv_91F02_inactivate[:, baseInds, 0],1)-np.nanmean(tracesAllIndv_91F02_inactivate[:, avgInds, 0],1),
            np.nanmean(tracesAllIndv_91F02_inactivate[:, baseInds, 2])-np.nanmean(tracesAllIndv_91F02_inactivate[:, avgInds, 2],1)))
    avg91F02 = np.nanmean(all91F02)
    ax.plot(5, np.nanmean(avg91F02),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    H1, p1 = stats.levene(all18D07,all91F02,center='mean')
    H2, p2 = stats.levene(all18D07,allEmpty,center='mean')
    H3, p3 = stats.levene(all91F02, allEmpty,center='mean')

    print(p1)
    print(p2)
    print(p3)

    plt.suptitle('inactivation light response\n mean empty inactivate='+str(round_to_significant_digit(avgEmpty,2))+
        '\n mean 18D07 ='+str(round_to_significant_digit(avg18D07,2))+
        '\n mean 91F02 inactivate='+str(round_to_significant_digit(avg91F02,3))+
        '\n levene''s test 18D07 vs. 91F02 p='+str(round_to_significant_digit(p1,2))+', H='+str(round_to_significant_digit(H1,2))+
        '\n levene''s test 18D07 vs. empty p='+str(round_to_significant_digit(p2,2))+', H='+str(round_to_significant_digit(H2,2))+
        '\n levene''s test 91F02 vs. empty p='+str(round_to_significant_digit(p3,2))+', H='+str(round_to_significant_digit(H3,2)),
        color=const.axisColor,fontsize=const.fontSize_angPair)

    # plot a representation of the p-value (ns or *, **, ***)
    yMax = 5.75
    sft = 0.1
    xLocComp = [[1.5,7],[1.5,13]]

    # configure the axes and plot color, etc.
    ax.set_facecolor(const.figColor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(const.axisColor)
    ax.spines['bottom'].set_color(const.axisColor)
    ax.tick_params(direction='in', length=5, width=0.5)
    ax.tick_params(axis='y',colors=const.axisColor)
    ax.set_ylabel('average steady state deflection',color=const.axisColor,fontsize=const.fontSize_axis)
    # configure the y-axis
    ax.set_ylim([-5,8])
    ax.set_yticks([-4, -2, 0, 2, 4, 6])
    ax.spines['left'].set_bounds(-4, 6) #do not extend y-axis line beyond ticks
    ## configure the x-axis
    ax.set_xlim([-1,6])
    ax.set_xticks([1,3,5])
    ax.tick_params(axis='x',colors=const.axisColor,rotation=30)
    ax.spines['bottom'].set_bounds(1, 5) #do not extend x-axis line 4 ticks
    xlabels = ['emptyGAL4','18D07','91F02']
    ax.set_xticklabels(xlabels,rotation=30,horizontalalignment='right')

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath +'inctivationLightResponseComparison.png'
        print('Saving figure here: ' + savepath)
        fig.savefig(savepath, facecolor=fig.get_facecolor())

        savepath = figPath +'inctivationLightResponseComparison.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())


def plot_light_response_quant_inactivation_leftRightSeparate(savefig=0):

    fig, ax = plt.subplots(1,1,facecolor=const.figColor,figsize=(4,6))
    plt.suptitle('inactivation light response',
     color=const.axisColor,fontsize=const.fontSize_angPair)

    extraText = '_withLight_' #no light is just a flat line, no need to quantify here
    expt1 = 'emptyGAL4_inactivate'
    ang_fn = const.savedDataDirectory+expt1+extraText+'lightTracesIndv.npy'
    tracesAllIndv_emptyGAL4_inactivate = np.load(ang_fn)
    expt2 = '18D07_inactivate'
    ang_fn = const.savedDataDirectory+expt2+extraText+'lightTracesIndv.npy'
    tracesAllIndv_18D07_inactivate = np.load(ang_fn)
    expt3 = '91F02_inactivate'
    ang_fn = const.savedDataDirectory+expt3+extraText+'lightTracesIndv.npy'
    tracesAllIndv_91F02_inactivate = np.load(ang_fn)

    avgIndsStart = const.framerate*(const.activateStart+3)
    avgIndsStop = const.framerate*(const.activateStart+5)
    avgInds = range(avgIndsStart, avgIndsStop,1)  #steady-state, last 2 seconds of 5 s light pulse

    for ang in [0,2]: #only plot for 2nd segments
        #plot emptyGAL4_inactivate
        flyAvgTraces = tracesAllIndv_emptyGAL4_inactivate[:, avgInds, ang]
        flyAvgDeflections = np.nanmean(flyAvgTraces,1)
        jitterSft = 0.25
        ones = np.ones(np.shape(flyAvgDeflections)[0])
        jitter = jitterSft*np.random.rand(np.shape(flyAvgDeflections)[0])-jitterSft/2
        ax.plot((ang+1)*ones+jitter, flyAvgDeflections,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
        ax.plot((ang+1), np.nanmean(flyAvgDeflections),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

        #plot 18D07_inactivate
        flyAvgTraces = tracesAllIndv_18D07_inactivate[:, avgInds, ang]
        flyAvgDeflections = np.nanmean(flyAvgTraces,1)
        jitterSft = 0.25
        ones = np.ones(np.shape(flyAvgDeflections)[0])
        jitter = jitterSft*np.random.rand(np.shape(flyAvgDeflections)[0])-jitterSft/2
        ax.plot((ang+6)*ones+jitter, flyAvgDeflections,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
        ax.plot((ang+6), np.nanmean(flyAvgDeflections),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

        #plot 91F02_inactivate
        flyAvgTraces = tracesAllIndv_91F02_inactivate[:, avgInds, ang]
        flyAvgDeflections = np.nanmean(flyAvgTraces,1)
        jitterSft = 0.25
        ones = np.ones(np.shape(flyAvgDeflections)[0])
        jitter = jitterSft*np.random.rand(np.shape(flyAvgDeflections)[0])-jitterSft/2
        ax.plot((ang+11)*ones+jitter, flyAvgDeflections,marker='.',linestyle='None',color=const.markerColor_diffIndv, markersize=const.markerTuningIndv,alpha=const.transparencyAntTrace)
        ax.plot((ang+11), np.nanmean(flyAvgDeflections),marker='.',linestyle='None',color=const.markerColor_diffAvg, markersize=const.markerTuningAvg+1)

    flyAvgDeflectionsR = np.nanmean(tracesAllIndv_emptyGAL4_inactivate[:, avgInds, 0],1)
    flyAvgDeflectionsL = np.nanmean(tracesAllIndv_emptyGAL4_inactivate[:, avgInds, 2],1)
    absEmpty = abs(np.append(flyAvgDeflectionsR,flyAvgDeflectionsL))

    flyAvgDeflectionsR = np.nanmean(tracesAllIndv_18D07_inactivate[:, avgInds, 0],1)
    flyAvgDeflectionsL = np.nanmean(tracesAllIndv_18D07_inactivate[:, avgInds, 2],1)
    abs18D07 = abs(np.append(flyAvgDeflectionsR,flyAvgDeflectionsL))

    flyAvgDeflectionsR = np.nanmean(tracesAllIndv_91F02_inactivate[:, avgInds, 0],1)
    flyAvgDeflectionsL = np.nanmean(tracesAllIndv_91F02_inactivate[:, avgInds, 2],1)
    abs91F02 = abs(np.append(flyAvgDeflectionsR,flyAvgDeflectionsL))

    result1 = stats.mannwhitneyu(absEmpty, abs18D07, alternative='two-sided') #difference between 2nd antennal segments
    pVals1 = [result1.pvalue]
    print('empty vs. 18D07 mannwhitneyu pvalue='+str(pVals1))
    result2 = stats.mannwhitneyu(absEmpty, abs91F02, alternative='two-sided') #difference between 2nd antennal segments
    pVals2 = [result2.pvalue]
    print('empty vs. 18D07 mannwhitneyu pvalue='+str(pVals2))
    pVals = pVals1+pVals2

    ksTest_18D07 = stats.kstest(abs18D07,'norm', alternative='two-sided')
    print('kstest for 18D07 light deflection norm pval=: '+str(ksTest_18D07))
    ksTest_91F02 = stats.kstest(abs91F02,'norm', alternative='two-sided')
    print('kstest for 91F02 light deflection norm pval=: '+str(ksTest_91F02))

    # plot result of mannwhitneyu between groups (indicate significane level)
    ax.text(5, 6.25, 'empty vs. 18D07: ' +str(round_to_significant_digit(result1.pvalue,2)), color='gray')
    ax.text(5,6, 'empty vs. 91F02: ' +str(round_to_significant_digit(result2.pvalue,2)), color='gray')

    ax.text(1, -4.25, '18D07 kstest pval=: ' +str(round_to_significant_digit(ksTest_18D07.pvalue,2)), color='gray')
    ax.text(1,-4.75, '91F02 ktest pval=: ' +str(round_to_significant_digit(ksTest_91F02.pvalue,2)), color='gray')

    # plot a representation of the p-value (ns or *, **, ***)
    yMax = 5.75
    sft = 0.1
    xLocComp = [[1.5,7],[1.5,13]]
    for ii in range(np.shape(pVals)[0]):
        yy = yMax-ii*sft
        ax.plot(xLocComp[ii],[yy,yy], color=const.axisColor, linewidth=1)
        if pVals[ii] < 0.001:
            mkr = '***'
        elif pVals[ii] < 0.01:
            mkr = '**'
        elif pVals[ii] < 0.05:
            mkr = '*'
        else: mkr = 'ns'
        ax.text(ii*4+3.5,yy+sft/10,mkr,color=const.axisColor,fontsize=const.fontSize_axis+1)

    # configure the axes and plot color, etc.
    ax.set_facecolor(const.figColor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(const.axisColor)
    ax.spines['bottom'].set_color(const.axisColor)
    ax.tick_params(direction='in', length=5, width=0.5)
    ax.tick_params(axis='y',colors=const.axisColor)
    ax.set_ylabel('average steady state deflection',color=const.axisColor,fontsize=const.fontSize_axis)
    # configure the y-axis
    ax.set_ylim([-5,6])
    ax.set_yticks([-5,0,5])
    ax.spines['left'].set_bounds(-5, 5) #do not extend y-axis line beyond ticks
    ## configure the x-axis
    ax.set_xlim([-1,14])
    ax.set_xticks([1,3,6,8,11,13])
    ax.tick_params(axis='x',colors=const.axisColor,rotation=30)
    ax.spines['bottom'].set_bounds(1, 13) #do not extend x-axis line 4 ticks
    xlabels = ['emptyGAL_R','emptyGAL4_L','18D07_R','18D07_L','91F02_R','91F02_L']
    ax.set_xticklabels(xlabels,rotation=30,horizontalalignment='right')

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath +'inctivationLightResponse.png'
        print('Saving figure here: ' + savepath)
        fig.savefig(savepath, facecolor=fig.get_facecolor())

        savepath = figPath +'inctivationLightResponse.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())

# Go through list of odor experiments and check if this experiment is one of them
def is_odor(expt):
    import loadNotes_mnOptoVelocityTuning as ln
    allExpts = eval('ln.notes_CS_odor')
    for noteInd, notes in enumerate(allExpts):
        flyExpt = notes['date']
        if(expt==flyExpt) :
            return True
        else:
            return False

# Go through set of antenna angles and omit (nan) any unusually large tracking errors
def detect_tracking_errors_raw_trace(exptNotes, expt, angs_all):
    numPreInds = int(exptNotes.pre_trial_time[0]*const.framerate)
    baselineInds = list(range(0, numPreInds))

    for exptNum in range(np.shape(angs_all)[0]): #go through all trials
        for ang in range(np.shape(angs_all)[2]): #for every antenna angle
            traces = angs_all[exptNum, :, ang] # grab raw angle for this trial

            baseline = np.nanmean(traces[baselineInds])
            tracesBaseSub = traces-baseline
            absTraces = np.absolute(tracesBaseSub)
            # detect and report large tracking errors (will want to quantify for flies and across datasets)
            trackingError = np.where(absTraces>const.TRACKING_ERROR_SIZE_CUTOFF)
            if np.shape(trackingError)[1] >= 1:
                print(str(trackingError) + ' <-- tracking error indices for expt ' + expt + ' trial ' + str(exptNum))
                indsForNan2 = [x + 1 for x in trackingError] # append this set to nan two inds for each error
                if np.any(np.where(np.squeeze(indsForNan2) == 480)): #should be a way to prevent this in above line? Not sure.
                    indsForNan2 = np.squeeze(indsForNan2)[:-1]
                nanInds = np.append(trackingError, indsForNan2)
                if np.any(np.where(np.squeeze(nanInds) == 480)): #one edge case (ind == length array) slips through above, why?
                    nanInds = nanInds[:-1]

                tracesBaseSub[nanInds] = np.nan # nan angles resulting from tracking error
                traces[nanInds] = np.nan
                angs_all[exptNum,nanInds,ang] = np.nan

    return angs_all

# detect active constants_active movements for one fly
# input: experiment name, TEST = 1 will plot traces with active movements overlaid
# output: matrix (ndarray) of active movement times along trace
# This is a good function to use in 'TEST' mode:
#   set TEST=trial of interest, or set TEST=-1 and manually go in and set trials to plot (deep in code) raw data, classification!
#   *however* - if in test mode, set FLIP_LEFT =0 (from 1 - typically flip for active movement analysis. But want non-flipped for raw traces!)
def get_active_movements(expt='2021_09_09_E1', cameraView='frontal',allowFlight = 1,TEST=0,importAnew=0,savefig=0):
    SAVE_TEST = 1 #if 1, will save a ton of images for raw trials
    if TEST:
        FLIP_LEFT = 0 # if 1, will flip left antennal movements (assuming symmetry - this is mostly for testing)
    else:
        FLIP_LEFT = 1 #(typical for grabbing active movements)
    exptNotes = get_mat_notes(expt)
    activate = get_activation_trials(expt)
    import loadNotes_mnOptoVelocityTuning as ln
    allExpts = eval('ln.notes_CS_odor')
    exptType = get_expt_type(expt) #grab class of experiment this is from

    if exptType == 'CS_odor':
        ODOR_EXPT = 1
    else: #the rest will be direction/velocity tuning
        ODOR_EXPT = 0

    yaxis_max = 25
    yaxis_min = -25

    numPreInds = int(exptNotes.pre_trial_time[0]*const.framerate)
    baselineInds = list(range(0, numPreInds))
    # Warning: hard-coding next line to *not* import traces anew (this will take a very long time, and is usually not what we want here - set to 1 in unique circumstances)
    [onsetAvg, onsetAllFlies, offsetAvg, offsetAllFlies, baseVals, tracesAvg, tracesAllIndv, baseAllFliesFullTrace] = get_cross_expt_lightResponse(exptType, cameraView,importAnew=0,allowFlight=allowFlight)
    [angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight] = getAntennaTracesAvgs_singleExpt(expt, cameraView, importAnew, allowFlight=allowFlight)
    # this gives us much more than we need (tracesAvg) but okay!
    [onsetAvg, onsetAllFlies, offsetAvg, offsetAllFlies, baseVals, tracesAvg, tracesAllIndv, baseAllFliesFullTrace] = get_single_expt_lightResponse(expt, cameraView, 0,allowFlight=allowFlight)

    if TEST:  # plot raw traces for all flies
        antIdx = [0,1,2,3]#[0,2]#  # analyze right 3rd segment to begin
        if ODOR_EXPT:
            dirIdxs = [3, 4, 7, 8]  # compare +45 directions, wind vs wind+odor
            ncols = 4
            velToPlot = -1
            nrows = int(np.shape(dirIdxs)[0])
            fig1, axAng = plt.subplots(nrows,ncols,facecolor=const.figColor,figsize=(12,12))
            fig1.suptitle(exptNotes.notes[0] + ' ' + exptNotes.date[0] + '_E' + str(exptNotes.expNumber[0][0]), fontsize=const.fsize_title-2, color=const.axisColor)
        else:
            dirIdxs = [2, 4, 6, 8, 10] # valve states (from original experiment)
            ncols = 4
            velToPlot = 200 #will plot traces from this example speed
            nrows = int(np.shape(dirIdxs)[0])
            fig2, axAngLight = plt.subplots(nrows,ncols,facecolor=const.figColor,figsize=(12,12))
            fig2.suptitle('Plotting for 200 cm/s here (light ON)'+'\n'+exptNotes.notes[0] + ' ' + exptNotes.date[0] + '_E' + str(exptNotes.expNumber[0][0]), fontsize=const.fsize_title-2, color=const.axisColor)

            fig1, axAng = plt.subplots(nrows,ncols,facecolor=const.figColor,figsize=(12,12))
            fig1.suptitle('Plotting for 200 cm/s here (no light)'+'\n'+exptNotes.notes[0] + ' ' + exptNotes.date[0] + '_E' + str(exptNotes.expNumber[0][0]), fontsize=const.fsize_title-2, color=const.axisColor)
    else:  # gather active traces for every condiditon (will filter or use as needed later)
        valveStates = exptNotes.valveState[:]
        if ODOR_EXPT:
            windDir = valveStates.str.get(0)
            speeds = []
        else:
            windDir = valveStates.str.get(0)#/2-1  # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
            windVel = exptNotes.velocity[:]
            windVel = windVel.str.get(0) #take values out of brackets
            speeds = np.unique(windVel)
        velToPlot = -1 #not plotting if not testing
        windDir = windDir.astype(int)
        windDirs = np.unique(windDir)
        dirIdxs = windDirs
        antIdx = list(range(np.shape(angs_all)[2]))

    # exclude peaks detected as a result of abrupt transitions (start of trace, wind on/light on, etc.)
    if ODOR_EXPT: #wind/odor on/offsets only
        windOnset = np.arange(const.odorWindOn*const.framerate-2, (const.odorWindOn*const.framerate)+const.excludeOnOffsetInds)
        windOffsetInds = np.arange(const.odorWindOff*const.framerate-10, (const.odorWindOff*const.framerate)+const.excludeOnOffsetInds)
        indsOnOffset = list(np.concatenate((0,windOnset,windOffsetInds),axis=None))
    else: #light and wind on/offsets
        lightOnsetInds = np.arange(const.activateStart*const.framerate-8,(const.activateStart*const.framerate)+const.excludeOnOffsetInds)
        lightOffsetInds = np.arange(const.activateStop*const.framerate-8, (const.activateStop*const.framerate)+const.excludeOnOffsetInds)
        windOnset = np.arange(const.windStart*const.framerate-2,(const.windStart*const.framerate)+const.excludeOnOffsetInds)
        windOffsetInds = np.arange((const.windStart+const.windTime)*const.framerate-10,(const.windStart+const.windTime)*const.framerate+const.excludeOnOffsetInds)
        indsOnOffset = list(np.concatenate((0,lightOnsetInds,lightOffsetInds,windOnset,windOffsetInds),axis=None))

    # prepare some things for filtering signal
    bb, aa = generate_butter_filter(const.framerate,fc=10)
    numBufSamples = 3 # number of samples to wrap beginning and end of trial
    prom = 1.5
    # initialize a large matrix to store active movement indices
    #AxBxC matrix with A=num trials, B=num valve states, C = #samples (length of trace)
    peaksAll = np.zeros([np.shape(angs_all)[0], np.shape(antIdx)[0], np.shape(angs_all)[1]]) #initialize empty peaks
    allDirs = np.zeros(np.shape(angs_all)[0])
    allVels = -np.ones(np.shape(angs_all)[0]) #will be all empty for CS_odor experiments

    for ind, dir in enumerate(dirIdxs):
        ct = 0
        # go through each experiment and grab traces for directions of interest
        thisDirTrials = exptNotes.index[(exptNotes.valveState == dir) == True].tolist()
        #plot active movement segments for each direction set (e.g. +45, +45 odor)
        if TEST:
            fig, axTrace = plt.subplots(1,np.shape(antIdx)[0],facecolor=const.figColor,figsize=(5,4))
        for angInd, ang in enumerate(antIdx):
            print('angInd= ' + str(angInd) + ' ang= ' +str(ang))
            thisDirTraces = angs_all[thisDirTrials, :, ang]
            avgTrace = np.nanmean(thisDirTraces,0)
            for exptNum in thisDirTrials:
                traces = angs_all[exptNum, :, ang] # grab raw angle for 3rd and 2nd seg this trial
                lightOn = activate[exptNum]
                tachFullSamp = exptNotes.tachometer[exptNum]
                tach = tachFullSamp[0::int(const.samplerate/const.framerate)+1]# downsample for plotting against antenna angles here

                 # assuming bilateral symmetry, flip left traces (HARD-CODED for 0,1,2,3 antenna parts)
                if FLIP_LEFT & (angInd > 1):
                    traces = -traces
                baseline = np.nanmean(traces[baselineInds])
                tracesBaseSub = traces-baseline
                if ODOR_EXPT:
                    vel = -1
                else:
                    vel = exptNotes.velocity[exptNum][0] # grab the speed for this trial
                allDirs[exptNum] = dir
                allVels[exptNum] = vel
                # take absolute value of trace
                absTraces = np.absolute(tracesBaseSub)

                # detect and report large tracking errors (will want to quantify for flies and across datasets)
                trackingError = np.where(absTraces>const.THRESH_ANG_CUTOFF)
                if np.shape(trackingError)[1] >= 1:
                    print(str(trackingError) + ' <-- tracking error indices for expt ' + expt + ' trial ' + str(exptNum))
                    indsForNan2 = [x + 1 for x in trackingError] # append this set to nan two inds for each error
                    if np.any(np.where(np.squeeze(indsForNan2) == 480)): #should be a way to prevent this in above line? Not sure.
                        indsForNan2 = np.squeeze(indsForNan2)[:-1]
                    nanInds = np.append(trackingError, indsForNan2)
                    tracesBaseSub[nanInds] = np.nan # nan angles resulting from tracking error
                    traces[nanInds] = np.nan
                    angs_all[exptNum,nanInds,ang] = np.nan

                # filter the data and detect peaks to classify active movementss
                #meanSubTracesBuff = np.concatenate([absTraces[0:numBufSamples], absTraces, absTraces[-numBufSamples:]])
                meanSubTracesBuff = np.concatenate([tracesBaseSub[0:numBufSamples], tracesBaseSub, tracesBaseSub[-numBufSamples:]])
                angFilt = signal.filtfilt(bb, aa, meanSubTracesBuff) # low-pass signal
                angFilt = angFilt[numBufSamples:-numBufSamples]    # trim the buffer off
                angFilt = np.absolute(angFilt)
                peaks, _ = signal.find_peaks(angFilt, prominence = prom) #,height=thresh_amp)#,threshold=thresh_ampFilt)
                peaksAll[exptNum, angInd, peaks] = 1
                # exclude any 'peaks' identified at light and wind on/offset
                if peaks.size != 0:
                    nullPeaks = [value for value, item in enumerate(peaks) if item in indsOnOffset]
                    peaks[nullPeaks] = 0
                    peaks = peaks[peaks>0]
                peaksAll[exptNum,angInd,indsOnOffset] = 0

                # plot raw data around each peak
                if TEST:
                    for ii, peak in enumerate(peaks):
                        start= int(peak-const.peakWindow)
                        stop = int(peak+const.peakWindow)
                        if (stop<const.numFrames) & (start>0): #take traces not overlapping with edge of trial (very beginning or very end)
                            axTrace[angInd].plot(tracesBaseSub[start:stop], color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                            if ODOR_EXPT:
                                plt.title(expt+', ang: '+ str(const.windOdorNames[dir-1]), color=const.axisColor, loc='left')
                            else:
                                plt.title(expt+', ang: '+ str(const.windDirNames[int(dir/2)-1]+' vel='+str(vel)), color=const.axisColor, loc='left')
                if ((TEST>0) & (exptNum == TEST)) | (TEST == -1):
                    fig, ax = plt.subplots(1,1,facecolor=[1,1,1], figsize=(10,8))

                    # draw wind and light activation stimulus bar
                    rectX = int(const.activateStart*const.framerate)
                    rectY = yaxis_min
                    rectWidLight = int(const.activateTime*const.framerate)
                    ax.add_patch(Rectangle((rectX,rectY),rectWidLight,const.stimBar_height,facecolor = const.color_activateColor))
                    ax.text(rectX+rectWidLight+10, rectY,
                        str(const.activateTime)+' s light on',color=const.color_activateColor,fontsize=const.fontSize_stimBar,horizontalalignment='left')
                    rectX = int(const.windStart*const.framerate)
                    rectWidWind = int(const.windTime*const.framerate)
                    rectY_wind = yaxis_min+1.5
                    ax.text(rectX+rectWidWind+10, rectY_wind,
                        str(const.windTime)+' s wind', fontsize=const.fontSize_stimBar,
                        horizontalalignment='left', color=const.axisColor)
                    ax.add_patch(Rectangle((rectX,rectY_wind), rectWidWind,const.stimBar_height,
                                 facecolor = const.axisColor))

                    ax.plot(angFilt, color='blue') #filtered trace (used for classification)
                    ax.text(0,18,'filtered trace', color='blue')
                    ax.plot(tach, color='gray') #filtered trace (used for classification)
                    ax.text(0,16,'tachometer', color='gray')

                    ax.plot(tracesBaseSub, color='black') #raw trace
                    ax.text(0,20,'raw angle (baseline-sub)', color='black')
                    ax.text(0,14,'active movement', color='orange')
                    if peaks.size != 0:
                        ax.plot(peaks, angFilt[peaks], 'x', color=const.peakMarker1)
                        ax.plot(peaks, tracesBaseSub[peaks], 'x', color=const.peakMarker2) #raw trace
                    ax.set_ylim(-30, 30)

                    plt.pause(0.001)
                    plt.show(block=False)

                    if lightOn == 1:
                        extraText = ' light on (inactivation)'
                    else: extraText = ''
                    if ODOR_EXPT:
                        plt.suptitle(expt+' trial '+str(exptNum+1)+' ang: '+ str(const.windOdorNames[dir-1])+' '+const.angPairNames[cameraView][ang]+ extraText, color='black')
                    else:
                        plt.suptitle(expt+' trial '+str(exptNum+1)+' ang: '+ str(const.windDirNames[int(dir/2)-1])+' '+const.angPairNames[cameraView][ang]+ extraText, color='black')
                    figTach, axTach =  plt.subplots(1,1,facecolor=[1,1,1], figsize=(10,8))
                    axTach.plot(tachFullSamp, color='gray')

                    if SAVE_TEST:
                        today = date.today()
                        dateStr = today.strftime("%Y_%m_%d")
                        figPath = const.savedFigureDirectory+str(dateStr)+'/'
                        if not os.path.isdir(figPath):
                            os.mkdir(figPath)
                        savepath = figPath + expt+ '_trial_'+str(exptNum+1)+' ang: '+ str(const.windOdorNames[dir-1])+' '+const.angPairNames[cameraView][ang] + '.pdf'
                        fig.savefig(savepath, facecolor=fig.get_facecolor())
                        savepath = figPath + expt+ '_trial_'+str(exptNum+1)+' ang: '+ str(const.windOdorNames[dir-1])+' '+const.angPairNames[cameraView][ang] + '.png'
                        fig.savefig(savepath, facecolor=fig.get_facecolor())
                        if angInd == 0:
                            savepath = figPath + expt+ '_trial_'+str(exptNum+1)+' ang: '+ str(const.windOdorNames[dir-1])+' '+const.angPairNames[cameraView][ang] + '_fullTach.pdf'
                            figTach.savefig(savepath, facecolor=fig.get_facecolor())
                            savepath = figPath + expt+ '_trial_'+str(exptNum+1)+' ang: '+ str(const.windOdorNames[dir-1])+' '+const.angPairNames[cameraView][ang] + '_fullTach.png'
                            figTach.savefig(savepath, facecolor=fig.get_facecolor())

                # plot raw traces for all flies and active movements (and only if specific velocity if not odor, to keep things simpler for viewing here)
                if TEST & (vel==velToPlot):
                    if lightOn == 1:
                        axAngLight[ind,0].plot(traces-baseline, linewidth=const.traceWidRaw,
                            color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                    else:
                        axAng[ind,0].plot(traces-baseline, linewidth=const.traceWidRaw,
                            color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                    if ODOR_EXPT:
                        axAng[ind,0].text(-int(const.activateAvgSt*const.framerate)*0.75, const.shiftYTraces,
                            str(const.windOdorNames[dir-1]) + const.degree_sign,
                            fontsize=const.fsize_velTuning_sub,horizontalalignment='center',
                            rotation=90, color=const.axisColor)
                    else:
                        axAng[ind,0].text(-int(const.activateAvgSt*const.framerate)*0.75, const.shiftYTraces,
                            str(const.windDirNames[ind]) + const.degree_sign,
                            fontsize=const.fsize_velTuning_sub,horizontalalignment='center',
                            rotation=90, color=const.axisColor)
                    # plot mean-subtracted traces
                    axAng[ind,1].plot(tracesBaseSub, linewidth=const.traceWidRaw,
                        color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                    # plot mean-subtracted traces
                    axAng[ind,2].plot(absTraces, linewidth=const.traceWidRaw,
                        color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                    # plot mean-subtracted traces
                    axAng[ind,3].plot(angFilt, linewidth=const.traceWidRaw,
                        color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                    # plot raster of identified peaks below the traces
                    if peaks.size != 0:
                        axAng[ind,3].plot(peaks, angFilt[peaks], 'x', color='white',linewidth=const.traceWidRaw, markersize=1)
                        pks = np.where(peaksAll[exptNum,angInd,:]>0)
                        peaksY = np.ones(np.shape(pks))*(-2)-ct*0.3
                        axAng[ind,3].plot(pks, peaksY,'.',color='white',linewidth=const.traceWidRaw,markersize=1)

                        ct += 1 # a tally of all the traces (for spacing in plot)
            baseline = np.nanmean(avgTrace[baselineInds])
            if TEST & ODOR_EXPT:
                axAng[ind,0].plot(avgTrace-baseline, linewidth=const.traceWidAvg,
                    color=const.colors_antAngs[ang])# alpha=const.transparencyAntTrace)

    tracesMeanSub = np.subtract(tracesAllIndv, tracesAvg)

    # set axis look
    if TEST:
        for col in range(ncols):
            for row in range(nrows):
                axAng[row,col].set_facecolor(const.figColor)
                axAng[row,col].set_ylim(yaxis_min, yaxis_max)
                axAng[row,col].axis('off')
        if ODOR_EXPT: axAng[0,0].text(0,15, 'raw and average',color=const.axisColor)
        else: axAng[0,0].text(0,15, 'raw traces',color=const.axisColor)
        axAng[0,1].text(0,15, 'baseline subtracted',color=const.axisColor)
        axAng[0,2].text(0,15, 'abs(baseline subtracted)',color=const.axisColor)
        axAng[0,3].text(0,15, '>threshold (of filtered trace)',color=const.axisColor)

    plt.show()
    plt.pause(0.001)
    plt.show(block=False)

    if savefig: # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        savepath = figPath + 'traces_noLight' + expt + '.png'
        fig1.savefig(savepath, facecolor=fig1.get_facecolor())
        savepath = figPath + 'activeTraces_noLight' + expt + '.pdf'
        fig1.savefig(savepath, facecolor=fig1.get_facecolor())

        savepath = figPath + 'traces_withLight' + expt + '.png'
        fig2.savefig(savepath, facecolor=fig2.get_facecolor())
        savepath = figPath + 'traces_withLight' + expt + '.pdf'
        fig2.savefig(savepath, facecolor=fig2.get_facecolor())

    return peaksAll, angs_all, allDirs, allVels


# Plot times of active movements as a raster for a single fly
def plot_active_movement_raster_single_fly_odor(expt='2021_09_09_E1',cameraView='frontal'):
    ODOR_EXPT = 1
    exptNotes = get_mat_notes(expt)
    peaksAll, allTraces, allDirs = get_active_movements(expt, cameraView='frontal',allowFlight=1,TEST=0)
    valveStates = exptNotes.valveState[:]
    if ODOR_EXPT:
        windDir = valveStates.str.get(0)
    else:
        windDir = valveStates.str.get(0)/2-1  # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)
    windDirs = np.unique(windDir)
    dirIdxs = windDirs
    antIdx = list(range(np.shape(allTraces)[2]))
    #set up figure
    fig, axRast = plt.subplots(1,1,facecolor=const.figColor,figsize=(5,4))
    axRast.set_facecolor(const.figColor)
    axRast.axis('off')
    # iterate through active movement time points and plot as a raster
    for ind, dir in enumerate(dirIdxs):
        ct = 0 # a tally of all the traces (for spacing in plot)
        # go through each experiment and grab traces for directions of interest
        thisDirTrials = exptNotes.index[(exptNotes.valveState == dir) == True].tolist()
        for angInd, ang in enumerate(antIdx):
            for exptNum in thisDirTrials:
                peaks = peaksAll[exptNum,angInd,:]
                if peaks.size != 0:
                    pks = np.where(peaksAll[exptNum,angInd,:]>0)
                    peaksY = np.ones(np.shape(pks))*(-2)-ct*0.3
                    axRast.plot(pks, peaksY,'.',color='white',linewidth=const.traceWidRaw,markersize=1)
                    ct = ct+1


# Plot times of active movements as a raster across flies for the wind-odor dataset
# This function is pretty hard-coded for the 'CS_odor' wind-odor data set.
def plot_active_movement_raster_cross_fly_odor(cameraView='frontal',importAnew=0,savefig = 0,secondSeg=1):
    expt='CS_odor'
    ODOR_EXPT = 1
    # get this experiment type list of experiments from notes
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)
    # set up figure
    numConds = 5 # no wind, ipsi, ipsi odor, contra, contra odor

    ct = [0,0,0,0,0] # a tally of all the traces (for spacing in plot)
    ctFlying = [0,0,0,0,0] # a tally of all the traces (for spacing in plot)
    ctPartialFlying = [0,0,0,0,0] # a tally of all the traces (for spacing in plot)
    numActive_noOdor_nonflying = []
    numActive_noOdor_flying = []
    numActive_odor_nonflying = []
    numActive_odor_flying = []

    # set up a set of arrays to tally up active movements
    pkTally = np.zeros([3,numConds,480],int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)
    # iterate through every fly and plot its raster
    for noteInd, notes in enumerate(allExpts):
        plotColor=const.singleFlyRasterColors[noteInd]
        flyExpt = notes['date']
        exptNotes = get_mat_notes(flyExpt)

        # Save data structures for faster plowwing
        saved_peaksAll_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'peaksAll.npy'
        saved_allTraces_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allTraces.npy'
        saved_allDirs_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allDirs.npy'
        saved_allVels_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allVels.npy'
        saved_isFlying_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'isFlying.npy'
        saved_flyingPercent_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'flyingPercent.npy'
        # check if the notes have been saved previously, then load
        if (importAnew==0) & (os.path.isfile(saved_peaksAll_fn)==True):
            # load previously saved active movement data
            peaksAll = np.load(saved_peaksAll_fn)
            allTraces = np.load(saved_allTraces_fn)
            allDirs = np.load(saved_allDirs_fn)
            allVels = np.load(saved_allVels_fn)
            isFlying = np.load(saved_isFlying_fn)
            flyingPercent = np.load(saved_flyingPercent_fn)
        else: # load notes from scratch (first time analyzing or re-analyzing)
            print('Importing active movements for this experiment: '+flyExpt)
            # import active movements
            peaksAll, allTraces, allDirs, allVels = get_active_movements(flyExpt, cameraView,allowFlight=1,TEST=0)
            isFlying, flyingPercent,overThresh = getFlightsSingleExpt(flyExpt, cameraView, importAnew)
            # save active movements
            np.save(saved_peaksAll_fn, peaksAll)
            np.save(saved_allTraces_fn, allTraces)
            np.save(saved_allDirs_fn, allDirs)
            np.save(saved_allVels_fn, allVels)
            np.save(saved_isFlying_fn, isFlying)
            np.save(saved_flyingPercent_fn, flyingPercent)

        valveStates = exptNotes.valveState[:]
        if ODOR_EXPT:
            windDir = valveStates.str.get(0)
        else:
            windDir = valveStates.str.get(0)/2-1  # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
        windDir = windDir.astype(int)
        windDirs = np.unique(windDir)
        dirIdxs = windDirs  # typically 0 (no wind), 3 (-45), 4 (-45 odor), 7 (+45), and 8 (+45 odor)
        if secondSeg:
            antIdxs = [0, 2]  # look at third segments of right and left antenna
        else:
            antIdxs = [1, 3]  # look at third segments of right and left antenna

        if noteInd == 0:
            numDirs = np.shape(dirIdxs)[0]
            numTrials = np.zeros([3,numDirs],int) # NxM (N=nonflying,flying, someFlight and M = 5 conditions (no wind, ipsi wind, etc.)) a tally of how many trials have active movements
            # compute total number of active movements during wind
            flyExpt = allExpts[0]['date']
            exptNotes = get_mat_notes(flyExpt)
            stimStart = exptNotes.pre_trial_time[0][0]
            stimStop = exptNotes.pre_trial_time[0][0]+exptNotes.trial_time_wind[0][0]
            windStart = stimStart*const.framerate
            windStop = stimStop*const.framerate
            windInds = np.arange(windStart, windStop)
        # iterate through active movement time points and plot as a raster
        for ind, dir in enumerate(dirIdxs):
            # go through each experiment and grab traces for directions of interest (this does not appear to be efficient but accurate...)
            thisDirTrials = exptNotes.index[(exptNotes.valveState == dir) == True].tolist()
            for antInd, ant in enumerate(antIdxs):
                if dir == 0:  # begin the elaborate way of placing each raster in ipsi/contra/no wind plot!
                    axInd = 0
                elif ((ant == 1) & (dir == 7)) | ((ant == 3) & (dir == 3)): # ipsi wind (R and +45 or L and -45)
                    axInd = 1
                elif ((ant == 1) & (dir == 8)) | ((ant == 3) & (dir == 4)): # ipsi odor (R and +45 or L and -45)
                    axInd = 2
                elif ((ant == 1) & (dir == 3)) | ((ant == 3) & (dir == 7)): # contra wind (R and -45 or L and +45)
                    axInd = 3
                elif ((ant == 1) & (dir == 4)) | ((ant == 3) & (dir == 8)): # contra wind odor (R and -45 or L and +45)
                    axInd = 4
                elif ((ant == 0) & (dir == 7)) | ((ant == 2) & (dir == 3)): # ipsi wind (R and +45 or L and -45)
                    axInd = 1
                elif ((ant == 0) & (dir == 8)) | ((ant == 2) & (dir == 4)): # ipsi odor (R and +45 or L and -45)
                    axInd = 2
                elif ((ant == 0) & (dir == 3)) | ((ant == 2) & (dir == 7)): # contra wind (R and -45 or L and +45)
                    axInd = 3
                elif ((ant == 0) & (dir == 4)) | ((ant == 2) & (dir == 8)): # contra wind odor (R and -45 or L and +45)
                    axInd = 4
                else:  # this should never happen
                    print('Something wrong with antenna/direction indices in cross-fly raster function')
                for exptInd, exptNum in enumerate(thisDirTrials):
                    peaks = peaksAll[exptNum,ant,:]
                    #peaks = peaksAll[exptNum,antInd,:]
                    if peaks.size != 0:
                        if sum(peaks) > 0:
                            if isFlying[exptNum] == 0:
                                numTrials[0, axInd] += 1
                            if isFlying[exptNum] == 1:
                                numTrials[1, axInd] += 1
                            if isFlying[exptNum] == 0.5:
                                numTrials[2, axInd] += 1
                        #pks = np.where(peaksAll[exptNum,antInd,:] > 0)
                        pks = np.where(peaksAll[exptNum,ant,:] > 0)
                        #peaksY = np.ones(np.shape(pks))*(-2)-ct[axInd]*0.3
                        if np.size(np.squeeze(pks)) <= 1:
                            pksInWind = [value for value in list(pks) if value in windInds]
                        else:
                            pksInWind = list(set(np.squeeze(pks)) & set(windInds))
                        # plot and quantify state-dependent peaks (during flight, nonflying)
                        if isFlying[exptNum] == 1:
                            peaksY = np.ones(np.shape(pks))*(-2)-ctFlying[axInd]*0.3
                            #axRast[1,axInd].plot(pks, peaksY,',',color=plotColor,linewidth=const.traceWidRaw,markersize=1)
                            ctFlying[axInd] += 1

                        elif isFlying[exptNum]:
                            peaksY = np.ones(np.shape(pks))*(-2)-ctPartialFlying[axInd]*0.3
                            #axRast[2, axInd].plot(pks, peaksY, '.', color=plotColor, linewidth=const.traceWidRaw, markersize=2)
                            ctPartialFlying[axInd] += 1
                            #gather single-trial data here for # peaks *during wind* (for quantification of odor vs. no odor)
                            if (axInd == 1) | (axInd == 3):  # may not be most efficient but works - trial-by-trial # movements during wind
                                numActive_noOdor_flying = np.append(numActive_noOdor_flying, np.size(pksInWind))
                            elif (axInd == 2) | (axInd == 4):
                                numActive_odor_flying = np.append(numActive_odor_flying, np.size(pksInWind))
                        else:
                            peaksY = np.ones(np.shape(pks))*(-2)-ct[axInd]*0.3
                            #axRast[0,axInd].plot(pks, peaksY,'.',color=plotColor,linewidth=const.traceWidRaw,markersize=2)
                            ct[axInd] += 1
                            if (axInd == 1) | (axInd == 3):  # may not be most efficient but works - trial-by-trial # movements during wind
                                numActive_noOdor_nonflying = np.append(numActive_noOdor_nonflying, np.size(pksInWind))
                            elif (axInd == 2) | (axInd == 4):
                                numActive_odor_nonflying = np.append(numActive_odor_nonflying, np.size(pksInWind))

                        for ii,pk in enumerate(pks):
                            if isFlying[exptNum] == 1:
                                pkTally[1,axInd,pk] += 1
                            elif isFlying[exptNum]:
                                pkTally[2,axInd,pk] += 1
                            else:
                                pkTally[0,axInd,pk] += 1

    print('trials fully flying: '+str(ctFlying))
    print('trials partially flying: '+str(ctPartialFlying))
    print('trials nonflying: '+str(ct))

    bb, aa = generate_butter_filter(const.framerate,fc=7) #filter for spike histogram
    scaleHist = 4
    #xlimAll = [0, const.numFrames+50]
    #ylimAll = [-5, 15]
    ylimHist = [-5, 20]

    # make no odor vs. odor plot (separate flying and partial flying)
    # HARD-CODED for odor/no odor (combine ipsi and contra data here)
    fig3, ax_odorNoOdor = plt.subplots(1,4,facecolor=const.figColor,figsize=(12,8))
    flInd = 0 #just nonflight data for this comparison
    axInd =  0

    # plot no odor data (combine ipsi+contra)
    tally = pkTally[flInd,1,:]+pkTally[flInd,3,:]
    trialsNoOdor = numTrials[flInd,1]+numTrials[flInd,3]
    tallyNoOdor = (tally/trialsNoOdor)*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
    angFilt = signal.filtfilt(bb,aa,tallyNoOdor)*scaleHist # low-pass signals and scale
        # plot the histograms on top of each other
    ax_odorNoOdor[0].plot(np.arange(0,480), angFilt, color=const.axisColor)

    # plot odor data (combine ipsi+contra)
    tally = pkTally[flInd,2,:]+pkTally[flInd,4,:]
    trialsOdor = numTrials[flInd,2]+numTrials[flInd,4]
    tallyOdor = (tally/trialsOdor)*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
    angFilt = signal.filtfilt(bb,aa,tallyOdor)*scaleHist # low-pass signals and scale
        # plot the histograms on top of each other
    ax_odorNoOdor[1].plot(np.arange(0,480), angFilt, color=const.axisColor)

    # plot flight raster in same fashion as odor
    # plot odor data (combine ipsi+contra)
    flyingInd = 2
    tally = pkTally[flInd,1,:]+pkTally[flyingInd,3,:]
    trialsFlight = numTrials[flyingInd,1]+numTrials[flyingInd,3]
    tallyFlight = (tally/trialsFlight)*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
    angFilt = signal.filtfilt(bb,aa,tallyFlight)*scaleHist # low-pass signals and scale
        # plot the histograms on top of each other
    ax_odorNoOdor[2].plot(np.arange(0,480), angFilt, color=const.axisColor)

    # plot flight raster in same fashion as odor
    # plot odor data (combine ipsi+contra)
    flyingInd = 2
    tally = pkTally[flInd,2,:]+pkTally[flyingInd,4,:]
    trialsFlightOdor = numTrials[flyingInd,2]+numTrials[flyingInd,4]
    tallyFlightOdor = (tally/trialsFlightOdor)*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
    angFilt = signal.filtfilt(bb,aa,tallyFlightOdor)*scaleHist # low-pass signals and scale
    # plot the histograms on top of each other
    ax_odorNoOdor[3].plot(np.arange(0,480), angFilt, color=const.axisColor)

    rectX = int(const.windStart*const.framerate)
    rectWid = int(const.windTime*const.framerate)
    stimBar_height = const.stimBar_height*0.2
    rectY_wind = -1.5# yaxis_min+1.5
    ax_odorNoOdor[0].text(rectX, rectY_wind-stimBar_height-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(const.windTime)+' s wind', fontsize=const.fontSize_stimBar,
        horizontalalignment='left', color=const.axisColor)
    ax_odorNoOdor[0].add_patch(Rectangle((rectX,rectY_wind), rectWid,stimBar_height,
        facecolor = const.axisColor))
    ax_odorNoOdor[1].text(rectX, rectY_wind-stimBar_height-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(const.windTime)+' s odor', fontsize=const.fontSize_stimBar,
        horizontalalignment='left', color=const.axisColor)
    ax_odorNoOdor[1].add_patch(Rectangle((rectX,rectY_wind), rectWid,stimBar_height,
        facecolor = const.axisColor))

    ax_odorNoOdor[2].text(rectX, rectY_wind-stimBar_height-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(const.windTime)+' s wind', fontsize=const.fontSize_stimBar,
        horizontalalignment='left', color=const.axisColor)
    ax_odorNoOdor[2].add_patch(Rectangle((rectX,rectY_wind), rectWid,stimBar_height,
        facecolor = const.axisColor))
    ax_odorNoOdor[3].text(rectX, rectY_wind-stimBar_height-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(const.windTime)+' s odor', fontsize=const.fontSize_stimBar,
        horizontalalignment='left', color=const.axisColor)
    ax_odorNoOdor[3].add_patch(Rectangle((rectX,rectY_wind), rectWid,stimBar_height,
        facecolor = const.axisColor))

    # configure the plots
    ylimOdorNoOdor = [-1, 12]
    if axInd == 0:
        ax_odorNoOdor[0].set_title('no odor', color=const.axisColor)
        ax_odorNoOdor[1].set_title('odor', color=const.axisColor)
        ax_odorNoOdor[2].set_title('flight no odor', color=const.axisColor)
        ax_odorNoOdor[3].set_title('flight odor', color=const.axisColor)
        scaleHist = 2
        histShiftY = 5
        scaleWidth = 5
        scaleBarSize = 3 # absolute count?
        scaleX = const.numFrames-const.framerate-5
        ax_odorNoOdor[0].add_patch(Rectangle((scaleX,histShiftY),scaleWidth,scaleBarSize*scaleHist,
            facecolor = const.axisColor))
        #label scale bar
        ax_odorNoOdor[0].text(scaleX+scaleWidth*10,histShiftY+scaleBarSize/2,str(scaleBarSize)+' Hz',fontsize=const.fontSize_stimBar,color=const.axisColor)
    ax_odorNoOdor[0].set_facecolor(const.figColor)
    ax_odorNoOdor[0].set_ylim(ylimHist)
    ax_odorNoOdor[0].axis('off')
    ax_odorNoOdor[1].set_facecolor(const.figColor)
    ax_odorNoOdor[1].set_ylim(ylimHist)
    ax_odorNoOdor[1].axis('off')
    ax_odorNoOdor[2].set_facecolor(const.figColor)
    ax_odorNoOdor[2].set_ylim(ylimHist)
    ax_odorNoOdor[2].axis('off')
    ax_odorNoOdor[3].set_facecolor(const.figColor)
    ax_odorNoOdor[3].set_ylim(ylimHist)
    ax_odorNoOdor[3].axis('off')

    # violin plot version of the quantification is better than scatter!
    #  (individual trials with integer values only is hard to look at)
    fig5, ax_violin = plt.subplots(1,1,facecolor=const.figColor,figsize=(3,5))
    fig5.suptitle('Active movements during wind/odor',color=const.axisColor)
    xlabels = ['wind\nnonflying', 'odor\nnonflying', 'wind\nflying', 'odor\nflying']
    ax_violin.set_facecolor(const.figColor)
    data = [numActive_noOdor_nonflying, numActive_odor_nonflying, numActive_noOdor_flying, numActive_odor_flying]
    parts = ax_violin.violinplot(data,showmeans=False,showmedians=False,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(const.violinColor)
        pc.set_alpha(1)

    #means and standard deviation make more sense than median and quartiles with this data (integers between 0 and )
    #q1_1, med1, q3_1 = np.percentile(numActive_noOdor_nonflying,[25,50,75])
    q1_1 = np.nanmean(numActive_noOdor_nonflying)-np.nanstd(numActive_noOdor_nonflying)
    q3_1 = np.nanmean(numActive_noOdor_nonflying)+np.nanstd(numActive_noOdor_nonflying)

    #q1_2, med2, q3_2 = np.percentile(numActive_odor_nonflying,[25,50,75])
    q1_2 = np.nanmean(numActive_odor_nonflying)-np.nanstd(numActive_odor_nonflying)
    q3_2 = np.nanmean(numActive_odor_nonflying)+np.nanstd(numActive_odor_nonflying)

    #q1_3, med3, q3_3 = np.percentile(numActive_noOdor_flying,[25,50,75])
    q1_3 = np.nanmean(numActive_noOdor_flying)-np.nanstd(numActive_noOdor_flying)
    q3_3 = np.nanmean(numActive_noOdor_flying)+np.nanstd(numActive_noOdor_flying)

    #q1_4, med4, q3_4 = np.percentile(numActive_odor_flying,[25,50,75])
    q1_4 = np.nanmean(numActive_odor_flying)-np.nanstd(numActive_odor_flying)
    q3_4 = np.nanmean(numActive_odor_flying)+np.nanstd(numActive_odor_flying)

    # medians = [med1, med2, med3, med4] #median is not as helpful with this discretized data
    means = [np.mean(numActive_noOdor_nonflying),np.mean(numActive_odor_nonflying),np.mean(numActive_noOdor_flying),np.mean(numActive_odor_flying)]
    q1 = [q1_1,q1_2,q1_3,q1_4]
    q3 = [q3_1,q3_2,q3_3,q3_4]
    inds = np.arange(1,len(means)+1)
    ax_violin.scatter(inds,means,marker='o',color=const.light_gray, s=30,zorder=3)
    ax_violin.vlines(inds,q1,q3,color='k',linestyle='-', lw=3)

    # do some stats
    # plot result of ttest between groups (indicate significane level)
    result1 = stats.ttest_ind(numActive_noOdor_nonflying, numActive_odor_nonflying)
    result2 = stats.ttest_ind(numActive_noOdor_flying, numActive_odor_flying)
    result3 = stats.ttest_ind(numActive_noOdor_nonflying, numActive_noOdor_flying)
    pVals = [result1.pvalue, result2.pvalue, result3.pvalue]
    ax_violin.text(1.25, 9, 'wind vs. odor p-val: ' +str(round_to_significant_digit(result1.pvalue,2)), color=const.axisColor,fontsize=6)
    ax_violin.text(1.25, 8.7, 'wind vs. odor (flight) p-val: ' +str(round_to_significant_digit(result2.pvalue,2)), color=const.axisColor,fontsize=6)
    ax_violin.text(1.25, 8.4, 'wind vs. wind (flight) p-val: ' +str(round_to_significant_digit(result3.pvalue,2)), color=const.axisColor,fontsize=6)

    # print means, standard deviation, differences, and p-values for violin plots:
    print('wind_nonflying | odor_nonflying | wind_flying | odor_flying ')
    print('standard deviation:')
    print(q3)
    print('means:')
    print(means)
    print('sample sizes:')
    print([np.shape(numActive_noOdor_nonflying), np.shape(numActive_odor_nonflying), np.shape(numActive_noOdor_flying), np.shape(numActive_odor_flying)])
    print('flight vs. no flight:')
    result = stats.ttest_ind(numActive_noOdor_flying, numActive_noOdor_nonflying)
    print(str(np.nanmean(numActive_noOdor_flying)-np.nanmean(numActive_noOdor_nonflying)) + ' p-val='+str(round_to_significant_digit(result.pvalue,2)))
    print('odor vs. no odor:')
    result = stats.ttest_ind(numActive_odor_nonflying, numActive_noOdor_nonflying)
    print(str(np.nanmean(numActive_odor_nonflying)-np.nanmean(numActive_noOdor_nonflying)) + ' p-val='+str(round_to_significant_digit(result.pvalue,2)))


    yMax = 7.5
    sft = 0.3
    compInds = [[1,2],[3,4],[1,3]]
    sftAmt = [0,0,0.3]
    for ii in range(np.shape(pVals)[0]):
        yy = yMax-sftAmt[ii]
        #ax_violin.plot([1, ii+1],[yy,yy], color=const.axisColor, linewidth=1)
        print(compInds[ii][0])
        ax_violin.plot([compInds[ii][0], compInds[ii][1]],[yy,yy], color=const.axisColor, linewidth=1)
        if pVals[ii] < 0.001:
            mkr = '***'
        elif pVals[ii] < 0.01:
            mkr = '**'
        elif pVals[ii] < 0.05:
            mkr = '*'
        else: mkr = 'ns'
        ax_violin.text(compInds[ii][1]-0.5, yy-sftAmt[ii]+0.3, mkr,
                       color=const.axisColor, fontsize=const.fontSize_axis+1)

    # configure axes
    ax_violin.spines['right'].set_visible(False)
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['left'].set_color(const.axisColor)
    ax_violin.spines['bottom'].set_visible(False)
    ax_violin.tick_params(direction='in', length=5, width=0.5)
    ax_violin.tick_params(axis='y', colors=const.axisColor)
    ax_violin.set_ylabel('# movements during wind/odor', color=const.axisColor,
                         fontsize=const.fontSize_axis)
    # configure the y-axis
    ax_violin.set_ylim([-1, 9])
    ax_violin.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax_violin.spines['left'].set_bounds(0, 7)  # do not extend y-axis line beyond ticks
    ax_violin.set_xlim([0.5, 4.5])
    ax_violin.xaxis.set_ticks([])
    for ii, lab in enumerate(xlabels):
        ax_violin.text(ii+1, -1, lab, rotation=30,
                       horizontalalignment='right', color=const.axisColor)

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig:  # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        # save raster (cross-fly)
        savepath = figPath + 'activeRasterAllFlies_' + expt + '.png'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + 'activeRasterAllFlies_' + expt + '.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        # save histogram comparison
        savepath = figPath + 'activeHistAllFlies_' + expt + '.png'
        fig2.savefig(savepath, facecolor=fig2.get_facecolor())
        savepath = figPath + 'activeHistAllFlies_' + expt + '.pdf'
        fig2.savefig(savepath, facecolor=fig2.get_facecolor())
        # save histogram comparison
        savepath = figPath + 'odorNoOdor_' + expt + '.png'
        fig3.savefig(savepath, facecolor=fig3.get_facecolor())
        savepath = figPath + 'odorNoOdor_' + expt + '.pdf'
        fig3.savefig(savepath, facecolor=fig3.get_facecolor())

        # save histogram comparison
        savepath = figPath + 'windOdorViolin_' + expt + '.png'
        fig5.savefig(savepath, facecolor=fig5.get_facecolor())
        savepath = figPath + 'windOdorViolin_' + expt + '.pdf'
        fig5.savefig(savepath, facecolor=fig5.get_facecolor())


# Plot times of active movements as a raster across flies for an experiment type
# if expt='18D07_inactivate' or '91F02_inactivate', please specify plotInactivate = 1 (Gtacr inactivation) or 2 (no light trials)
def plot_active_movement_raster_cross_fly(expt='CS_activate',cameraView='frontal', plotInactivate=0,importAnew = 0,savefig = 0,secondSeg=0):

    # get this experiment type list of experiments from notes
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)
    # iterate through every fly and plot its raster
    for noteInd, notes in enumerate(allExpts):
        plotColor=const.singleFlyRasterColors[noteInd]
        flyExpt = notes['date']
        exptNotes = get_mat_notes(flyExpt)
        activate = get_activation_trials(flyExpt)
        if (np.sum(activate) != np.shape(activate)) & (expt != 'CS_activate'):
            lightCompare = 1 #will compare trials with and without light (e.g. for inactivation experiment)
            if plotInactivate == 0:
                print('Please enter ''plotInactivate=1 (for inactvation data with light) or plotInactivate=2 (for inactvation data with no light)')
                return

        # Save data structures for faster plotting
        if secondSeg == 1:
            antIdxs = [0,2] # look at third segments of right and left antenna (can also compare [0,2] to look at second segment)
            segmentText = 'secondSeg_'
        else: #default is to analyze third segment data
            antIdxs = [1,3] # look at third segments of right and left antenna (can also compare [0,2] to look at second segment)
            segmentText = 'thirdSeg_'
        saved_peaksAll_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+segmentText+'peaksAll.npy'
        saved_allTraces_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+segmentText+'allTraces.npy'
        saved_allDirs_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+segmentText+'allDirs.npy'
        saved_allVels_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+segmentText+'allVels.npy'
        saved_isFlying_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+segmentText+'isFlying.npy'
        saved_flyingPercent_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+segmentText+'flyingPercent.npy'
        save_overThresh_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+segmentText+'overThresh.npy'

        # check if the notes have been saved previously, then load
        if ((importAnew==0) & (os.path.isfile(saved_peaksAll_fn)==True)):
            # load previously saved active movement data
            peaksAll = np.load(saved_peaksAll_fn)
            allTraces = np.load(saved_allTraces_fn)
            allDirs = np.load(saved_allDirs_fn)
            allVels = np.load(saved_allVels_fn)
            isFlying = np.load(saved_isFlying_fn)
            flyingPercent = np.load(saved_flyingPercent_fn)
            overThresh = np.load(save_overThresh_fn)
        else: # load notes from scratch (first time analyzing or re-analyzing)
            print('Importing active movements for this experiment: '+flyExpt)
            # import active movements
            isFlying, flyingPercent, overThresh = getFlightsSingleExpt(flyExpt, cameraView, importAnew)
            peaksAll, allTraces, allDirs, allVels = get_active_movements(flyExpt, cameraView,allowFlight=1,TEST=0,importAnew=0);
            isFlying, flyingPercent, overThresh = getFlightsSingleExpt(flyExpt, cameraView, importAnew=0)

            # save active movements
            np.save(saved_peaksAll_fn, peaksAll)
            np.save(saved_allTraces_fn, allTraces)
            np.save(saved_allDirs_fn, allDirs)
            np.save(saved_allVels_fn, allVels)
            np.save(saved_isFlying_fn, isFlying)

            np.save(saved_flyingPercent_fn, flyingPercent)
            np.save(save_overThresh_fn, overThresh)

        # determine directions of wind we're plotting
        valveStates = exptNotes.valveState[:]
        windDir = valveStates.str.get(0)
        windDir = windDir.astype(int)
        windDirs = np.unique(windDir)
        dirIdxs = windDirs # typically 0 (no wind), 3 (-45), 4 (-45 odor), 7 (+45), and 8 (+45 odor)
        # determine how many speeds of wind were presented and should be plotted
        windVel = exptNotes.velocity[:]
        windVel = windVel.str.get(0) #take values out of brackets
        speeds = np.unique(windVel)

        if noteInd == 0: # set up the figure and some variables at start of iteration
            numDirs = np.shape(dirIdxs)[0]
            numVels = np.shape(speeds)[0]
            fig_nonflying, axRast_nonflying = plt.subplots(numDirs,numVels,facecolor=const.figColor,figsize=(7,10)) #3 columns for nonflying, flight, and partial flight trials
            fig_flying, axRast_flying = plt.subplots(numDirs,numVels,facecolor=const.figColor,figsize=(7,10)) #3 columns for nonflying, flight, and partial flight trials
            fig_someFlight, axRast_someFlight = plt.subplots(numDirs,numVels,facecolor=const.figColor,figsize=(7,10)) #3 columns for nonflying, flight, and partial flight trials

            numTrials_nonflying = np.zeros([numDirs, numVels], int) # [0,0,0,0,0] # a tally of all the traces (for spacing in plot)
            numTrials_flying = np.zeros([numDirs, numVels], int) # a tally of all the traces (for spacing in plot)
            numTrials_someFlight = np.zeros([numDirs, numVels], int) # a tally of all the traces (for spacing in plot)
            # set up a set of arrays to tally up active movements
            pkTally_nonflying = np.zeros([numDirs, numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)
            pkTally_flying = np.zeros([numDirs, numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)
            pkTally_someFlight = np.zeros([numDirs, numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)
            pkTally_flightPerMovement = np.zeros([numDirs, numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)

            numTrials_nonflying_allFlies = np.zeros([np.shape(allExpts)[0], numDirs, numVels], int) # [0,0,0,0,0] # a tally of all the traces (for spacing in plot)
            numTrials_flying_allFlies = np.zeros([np.shape(allExpts)[0], numDirs, numVels], int) # a tally of all the traces (for spacing in plot)
            numTrials_someFlight_allFlies = np.zeros([np.shape(allExpts)[0], numDirs, numVels], int) # a tally of all the traces (for spacing in plot)
            pkTally_nonflying_allFlies = np.zeros([np.shape(allExpts)[0], numDirs, numVels, 480], int)
            pkTally_flying_allFlies = np.zeros([np.shape(allExpts)[0], numDirs, numVels, 480], int)
            pkTally_someFlight_allFlies = np.zeros([np.shape(allExpts)[0], numDirs, numVels, 480], int)
            pkTally_flightPerMovement_allFlies = np.zeros([np.shape(allExpts)[0], numDirs, numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)

            numTrials = np.zeros([numDirs,numVels],int) # a tally of how many trials have active movements

        # iterate through active movement time points and plot as a raster
        for dir_ind, dir in enumerate(dirIdxs):
            # go through each experiment and grab traces for directions of interest
            thisDirTrials = exptNotes.index[(exptNotes.valveState == dir) == True].tolist()
            for antInd, ant in enumerate(antIdxs):
                for trialNum in thisDirTrials:
                    vel = allVels[trialNum]  # grab the speed, plot accordingly!
                    vel_ind = np.squeeze(np.where(vel==speeds))

                    if (plotInactivate==0) | ((plotInactivate==1) & activate[trialNum]) | ((plotInactivate==2) & ~activate[trialNum]):
                        peaks = peaksAll[trialNum,ant,:]

                        if antInd > 1:  # flip left antenna indices (onto right)
                            if dir_ind == 0: dir_ind_IC = 4
                            elif dir_ind == 1: dir_ind_IC = 3
                            elif dir_ind == 2: dir_ind_IC = 2  # no change for frontal wind
                            elif dir_ind == 3: dir_ind_IC = 1
                            elif dir_ind == 4: dir_ind_IC = 0  # and leave 2 (0 deg) alone!
                        else:  # leave right alone - already in ipsi/contra organization
                            dir_ind_IC = dir_ind
                        if peaks.size != 0: #& (plotInactivate == 0 | ((plotInactivate==1) & activate[trialNum]) | ((plotInactivate==2) & ~activate[trialNum])):
                            if sum(peaks) > 0:
                                numTrials[dir_ind_IC, vel_ind] += 1
                            pks = np.where(peaksAll[trialNum,ant,:]>0)
                            if isFlying[trialNum] == 1:
                                #print('isFlying!')
                                peaksY = np.ones(np.shape(pks))*(-2)-numTrials_flying[dir_ind_IC, vel_ind]*0.3
                                axRast_flying[dir_ind_IC,vel_ind].plot(pks, peaksY,',',color=plotColor,linewidth=const.traceWidRaw,markersize=1)
                                numTrials_flying[dir_ind_IC,vel_ind] += 1
                                numTrials_flying_allFlies[noteInd, dir_ind_IC,vel_ind] += 1
                            elif isFlying[trialNum]:
                                #print('partial flight')
                                peaksY = np.ones(np.shape(pks))*(-2)-numTrials_someFlight[dir_ind_IC, vel_ind]*0.3
                                axRast_someFlight[dir_ind_IC,vel_ind].plot(pks, peaksY,',',color=plotColor,linewidth=const.traceWidRaw,markersize=1)
                                numTrials_someFlight[dir_ind_IC,vel_ind] += 1
                                numTrials_someFlight_allFlies[noteInd, dir_ind_IC,vel_ind] += 1
                            else: #nonflying
                                peaksY = np.ones(np.shape(pks))*(-2)-numTrials_nonflying[dir_ind_IC, vel_ind]*0.3
                                axRast_nonflying[dir_ind_IC,vel_ind].plot(pks, peaksY,',',color=plotColor,linewidth=const.traceWidRaw,markersize=1)
                                numTrials_nonflying[dir_ind_IC,vel_ind] += 1
                                numTrials_nonflying_allFlies[noteInd,dir_ind_IC,vel_ind] += 1
                            for ii,pk in enumerate(pks):
                                if isFlying[trialNum] == 1:
                                    pkTally_flying[dir_ind_IC,vel_ind,pk] += 1
                                    pkTally_flying_allFlies[noteInd,dir_ind_IC,vel_ind,pk] += 1 #tally up for each fly
                                elif isFlying[trialNum]:
                                    pkTally_someFlight[dir_ind_IC,vel_ind,pk] += 1
                                    pkTally_someFlight_allFlies[noteInd,dir_ind_IC,vel_ind,pk] += 1 #tally up for each fly
                                else:
                                    pkTally_nonflying[dir_ind_IC,vel_ind,pk] += 1
                                    pkTally_nonflying_allFlies[noteInd,dir_ind_IC,vel_ind,pk] += 1 #tally up for each fly
                                # additional count = for *each* movement: is it during flight or not
                                # (above tallies classify whole trials as fully or partially flying)
                                if overThresh[trialNum,pk].any():
                                    pkTally_flightPerMovement[dir_ind_IC,vel_ind,pk] += 1
                                    pkTally_flightPerMovement_allFlies[noteInd,dir_ind_IC,vel_ind,pk] += 1

    # Plot histogram beneath raster, and configure the subplots at the same time
    bb, aa = generate_butter_filter(const.framerate,fc=5) #filter for spike histogram
    if plotInactivate == 0:
        fig_nonflying.suptitle(expt+' nonflying trials',color=const.axisColor, horizontalalignment='center')
        fig_flying.suptitle(expt+' flying trials',color=const.axisColor, horizontalalignment='center')
        fig_someFlight.suptitle(expt+' partially flying trials',color=const.axisColor, horizontalalignment='center')
    else:
        if plotInactivate == 1: actText = ' light (Gtacr inactivation)'
        else: actText = ' no light (no Gtacr inactivation)'
        fig_nonflying.suptitle(expt+' nonflying trials ' + actText,color=const.axisColor, horizontalalignment='center')
        fig_flying.suptitle(expt+' flying trials ' + actText,color=const.axisColor, horizontalalignment='center')
        fig_someFlight.suptitle(expt+' partially flying trials ' + actText,color=const.axisColor, horizontalalignment='center')
    scaleHist = 2
    histShiftY = 12
    scaleWidth = 3
    scaleBarSize = 3 # absolute count?
    scaleX = const.numFrames # x-position of scale bar (movements/sec)
    scaleY = -20
    maxAllCt = -np.max([numTrials_nonflying, numTrials_flying, numTrials_someFlight])
    ylimAll = [maxAllCt*0.3-20, 0]
    xlimAll = [0, const.numFrames+50]
    # light activation stimulus bar params
    ymin_nonflying = -np.max(numTrials_nonflying)*0.3-histShiftY-4
    ymin_flying = -np.max(numTrials_flying)*0.3-histShiftY-4
    ymin_someFlight = -np.max(numTrials_someFlight)*0.3-histShiftY-4
    rectHeight = const.stimBar_height
    rectX_light = int(const.activateStart*const.framerate)-1  # -1 to align with 0 indexing
    rectY_light_base = rectHeight*2-rectHeight*4  # add some more to accomodate wind stimulus (above)
    rectWid_light = int(const.activateTime*const.framerate)
    # wind stimulus bar params
    rectX_wind = int(const.windStart*const.framerate)-1  # -1 to align with 0 indexing
    rectY_wind_base = rectHeight*2
    rectWid_wind = int(const.windTime*const.framerate)
    for dir_ind, dir in enumerate(dirIdxs):
        for vel_ind, vel in enumerate(speeds):
            # for nonflying
            tally = (pkTally_nonflying[dir_ind,vel_ind,:]/numTrials_nonflying[dir_ind,vel_ind])*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
            angFilt = signal.filtfilt(bb,aa,tally)*scaleHist # low-pass signals and scale
            axRast_nonflying[dir_ind,vel_ind].plot(np.arange(0,480), angFilt-np.max(numTrials_nonflying)*0.3-histShiftY,linestyle='-',color=const.axisColor,alpha=const.transparencyHist) #plot filtered spike count
            # configure the subplots here too
            axRast_nonflying[dir_ind,vel_ind].set_facecolor(const.figColor)
            axRast_nonflying[dir_ind,vel_ind].set_ylim(ylimAll)
            axRast_nonflying[dir_ind,vel_ind].set_xlim(xlimAll)
            axRast_nonflying[dir_ind,vel_ind].axis('off')

            # configure and plot histogram for flight
            tally = (pkTally_flying[dir_ind,vel_ind,:]/numTrials_flying[dir_ind,vel_ind])*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
            angFilt = signal.filtfilt(bb,aa,tally)*scaleHist # low-pass signals and scale
            axRast_flying[dir_ind,vel_ind].plot(np.arange(0,480), angFilt-np.max(numTrials_flying)*0.3-histShiftY,linestyle='-',color=const.axisColor,alpha=const.transparencyHist) #plot filtered spike count
            # configure the subplots here too
            axRast_flying[dir_ind,vel_ind].set_facecolor(const.figColor)
            axRast_flying[dir_ind,vel_ind].axis('off')
            axRast_flying[dir_ind,vel_ind].set_ylim(ylimAll)
            axRast_flying[dir_ind,vel_ind].set_xlim(xlimAll)

            # configure and plot histogram for flight
            tally = (pkTally_someFlight[dir_ind,vel_ind,:]/numTrials_someFlight[dir_ind,vel_ind])*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
            angFilt = signal.filtfilt(bb,aa,tally)*scaleHist # low-pass signals and scale
            axRast_someFlight[dir_ind,vel_ind].plot(np.arange(0,480), angFilt-np.max(numTrials_someFlight)*0.3-histShiftY,linestyle='-',color=const.axisColor,alpha=const.transparencyHist) #plot filtered spike count
            # configure the subplots here too
            axRast_someFlight[dir_ind,vel_ind].set_facecolor(const.figColor)
            axRast_someFlight[dir_ind,vel_ind].axis('off')
            axRast_someFlight[dir_ind,vel_ind].set_ylim(ylimAll)
            axRast_someFlight[dir_ind,vel_ind].set_xlim(xlimAll)

            if dir_ind == 0:
                axRast_nonflying[dir_ind,vel_ind].set_title(str(vel)+' cm/s',color=const.axisColor, horizontalalignment='center',fontsize=const.fsize_raster)
                axRast_flying[dir_ind,vel_ind].set_title(str(vel)+' cm/s',color=const.axisColor, horizontalalignment='left',fontsize=const.fsize_raster)
                axRast_someFlight[dir_ind,vel_ind].set_title(str(vel)+' cm/s',color=const.axisColor, horizontalalignment='right',fontsize=const.fsize_raster)
            if vel_ind == 0: #plot ipsi/contra row labels
                axRast_nonflying[dir_ind,vel_ind].text(0, 0,
                    const.windDirLabelsIC[dir_ind], fontsize=const.fsize_raster,
                    horizontalalignment='left', rotation=90, color=const.axisColor)
                axRast_flying[dir_ind,vel_ind].text(0, 0,
                    const.windDirLabelsIC[dir_ind], fontsize=const.fsize_raster,
                    horizontalalignment='left', rotation=90, color=const.axisColor)
                axRast_someFlight[dir_ind,vel_ind].text(0, 0,
                    const.windDirLabelsIC[dir_ind], fontsize=const.fsize_raster,
                    horizontalalignment='left', rotation=90, color=const.axisColor)
            if (vel_ind == 0) & (dir_ind == 0): # plot movements/sec scale bar
                axRast_nonflying[dir_ind,vel_ind].add_patch(Rectangle((scaleX,-np.max(numTrials_nonflying)*0.3-histShiftY),scaleWidth,scaleBarSize*scaleHist,
                    facecolor = const.axisColor))
                axRast_flying[dir_ind,vel_ind].add_patch(Rectangle((scaleX,-np.max(numTrials_flying)*0.3-histShiftY),scaleWidth,scaleBarSize*scaleHist,
                    facecolor = const.axisColor))
                axRast_someFlight[dir_ind,vel_ind].add_patch(Rectangle((scaleX,-np.max(numTrials_someFlight)*0.3-histShiftY),scaleWidth,scaleBarSize*scaleHist,
                    facecolor = const.axisColor))
                #label scale bar
                axRast_nonflying[dir_ind,vel_ind].text(scaleX+scaleWidth*10,-np.max(numTrials_nonflying)*0.3-histShiftY,str(scaleBarSize),fontsize=const.fontSize_stimBar,color=const.axisColor)
                axRast_flying[dir_ind,vel_ind].text(scaleX+scaleWidth*10,-np.max(numTrials_flying)*0.3-histShiftY,str(scaleBarSize),fontsize=const.fontSize_stimBar,color=const.axisColor)
                axRast_someFlight[dir_ind,vel_ind].text(scaleX+scaleWidth*10,-np.max(numTrials_someFlight)*0.3-histShiftY,str(scaleBarSize),fontsize=const.fontSize_stimBar,color=const.axisColor)
            # plot the wind/light stimulus regions below the data
            if dir_ind == 4:
                axRast_nonflying[dir_ind,vel_ind].add_patch(Rectangle((rectX_wind, ymin_nonflying-rectY_wind_base), rectWid_wind, rectHeight,
                    facecolor=const.axisColor))
                axRast_nonflying[dir_ind,vel_ind].text(rectX_wind+rectWid_wind+5, ymin_nonflying-rectY_wind_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                    str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                    color=const.axisColor)
                axRast_flying[dir_ind,vel_ind].add_patch(Rectangle((rectX_wind, ymin_flying-rectY_wind_base), rectWid_wind, rectHeight,
                    facecolor=const.axisColor))
                axRast_flying[dir_ind,vel_ind].text(rectX_wind+rectWid_wind+5, ymin_flying-rectY_wind_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                    str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                    color=const.axisColor)
                axRast_someFlight[dir_ind,vel_ind].add_patch(Rectangle((rectX_wind, ymin_someFlight-rectY_wind_base), rectWid_wind, rectHeight,
                    facecolor=const.axisColor))
                axRast_someFlight[dir_ind,vel_ind].text(rectX_wind+rectWid_wind+5, ymin_someFlight-rectY_wind_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                    str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                    color=const.axisColor)

            if (dir_ind == 4) & (plotInactivate != 2):
                axRast_nonflying[dir_ind,vel_ind].add_patch(Rectangle((rectX_light, ymin_nonflying-rectY_light_base), rectWid_light, rectHeight,
                    facecolor=const.color_activateColor))
                axRast_nonflying[dir_ind,vel_ind].text(rectX_light+rectWid_light+5, ymin_nonflying-rectY_light_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                    str(const.activateTime)+' s light',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                    color=const.color_activateColor)
                axRast_flying[dir_ind,vel_ind].add_patch(Rectangle((rectX_light, ymin_flying-rectY_light_base), rectWid_light, rectHeight,
                    facecolor=const.color_activateColor))
                axRast_flying[dir_ind,vel_ind].text(rectX_light+rectWid_light+5, ymin_flying-rectY_light_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                    str(const.activateTime)+' s light',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                    color=const.color_activateColor)
                axRast_someFlight[dir_ind,vel_ind].add_patch(Rectangle((rectX_light, ymin_someFlight-rectY_light_base), rectWid_light, rectHeight,
                    facecolor=const.color_activateColor))
                axRast_someFlight[dir_ind,vel_ind].text(rectX_light+rectWid_light+5, ymin_someFlight-rectY_light_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                    str(const.activateTime)+' s light',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                    color=const.color_activateColor)

    plt.show()
    plt.pause(0.001)
    plt.show(block=False)


    if plotInactivate == 1:
        extraText = 'withLight_'
    elif plotInactivate == 2:
        extraText = 'noLight_'
    else:
        extraText = ''

    saved_pkTally_nonflying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_nonflying.npy'
    saved_pkTally_flying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_flying.npy'
    saved_pkTally_someFlight_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_someFlight.npy'
    saved_pkTally_flightPerMovement_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_flightPerMovement.npy'
    saved_pkTally_nonflying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_nonflying_allFlies.npy'
    saved_pkTally_flying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_flying_allFlies.npy'
    saved_pkTally_someFlight_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_someFlight_allFlies.npy'
    saved_pkTally_flightPerMovement_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_flightPerMovement_allFlies.npy'

    numTrials_nonflying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_nonflying.npy'
    numTrials_flying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_flying.npy'
    numTrials_someFlight_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_someFlight.npy'
    numTrials_nonflying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_nonflying_allFlies.npy'
    numTrials_flying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_flying_allFlies.npy'
    numTrials_someFlight_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_someFlight_allFlies.npy'

    np.save(saved_pkTally_nonflying_fn, pkTally_nonflying)
    np.save(saved_pkTally_flying_fn, pkTally_flying)
    np.save(saved_pkTally_someFlight_fn, pkTally_someFlight)
    np.save(saved_pkTally_flightPerMovement_fn, pkTally_flightPerMovement)

    np.save(saved_pkTally_nonflying_allFlies_fn, pkTally_nonflying_allFlies)
    np.save(saved_pkTally_flying_allFlies_fn, pkTally_flying_allFlies)
    np.save(saved_pkTally_someFlight_allFlies_fn, pkTally_someFlight_allFlies)
    np.save(saved_pkTally_flightPerMovement_allFlies_fn, pkTally_flightPerMovement_allFlies)

    np.save(numTrials_nonflying_fn, numTrials_nonflying)
    np.save(numTrials_flying_fn, numTrials_flying)
    np.save(numTrials_someFlight_fn, numTrials_someFlight)
    np.save(numTrials_nonflying_allFlies_fn, numTrials_nonflying_allFlies)
    np.save(numTrials_flying_allFlies_fn, numTrials_flying_allFlies)
    np.save(numTrials_someFlight_allFlies_fn, numTrials_someFlight_allFlies)

    if savefig: # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        # load data for active movements, flight
        if plotInactivate == 1:
            extraText = '_withLight_'
        elif plotInactivate == 2:
            extraText = '_noLight_'
        else:
            extraText = ''
        # save raster (cross-fly)
        savepath = figPath + expt + '_raster_direction_velocity_nonflying' + extraText+'.png'
        fig_nonflying.savefig(savepath, facecolor=fig_nonflying.get_facecolor())
        savepath = figPath + expt + '_raster_direction_velocity_nonflying' +  extraText+'.pdf'
        fig_nonflying.savefig(savepath, facecolor=fig_nonflying.get_facecolor())

        savepath = figPath + expt + '_raster_direction_velocity_flying' +  extraText+'.png'
        fig_flying.savefig(savepath, facecolor=fig_flying.get_facecolor())
        savepath = figPath + expt + '_raster_direction_velocity_flying' +  extraText+'.pdf'
        fig_flying.savefig(savepath, facecolor=fig_flying.get_facecolor())

        savepath = figPath + expt + '_raster_direction_velocity_partialFlights' +  extraText+'.png'
        fig_someFlight.savefig(savepath, facecolor=fig_someFlight.get_facecolor())
        savepath = figPath + expt + '_raster_direction_velocity_partialFlights' +  extraText+'.pdf'
        fig_someFlight.savefig(savepath, facecolor=fig_someFlight.get_facecolor())


# Plot times of active movements as a raster across flies for an experiment type
# This function is useful for looking at all inactivation data combined (across directions)
# if expt='18D07_inactivate' or '91F02_inactivate', please specify plotInactivate = 1 (Gtacr inactivation) or 2 (no light trials)
def plot_active_movement_raster_cross_fly_all_dir(expt='18D07_inactivate',cameraView='frontal', plotInactivate=1,importAnew = 0,savefig = 0,secondSeg=1):

    # get this experiment type list of experiments from notes
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)
    ysft_raster = 3#

    for noteInd, notes in enumerate(allExpts):
        plotColor=const.singleFlyRasterColors[noteInd]
        flyExpt = notes['date']
        exptNotes = get_mat_notes(flyExpt)
        activate = get_activation_trials(flyExpt)
        if (np.sum(activate) != np.shape(activate)) & (expt != 'CS_activate'):
            lightCompare = 1 #will compare trials with and without light (e.g. for inactivation experiment)
            if plotInactivate == 0:
                print('Please enter ''plotInactivate=1 (for inactvation data with light) or plotInactivate=2 (for inactvation data with no light)')
                return

        # Save data structures for faster plotting
        saved_peaksAll_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'peaksAll.npy'
        saved_allTraces_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allTraces.npy'
        saved_allDirs_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allDirs.npy'
        saved_allVels_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allVels.npy'
        saved_isFlying_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'isFlying.npy'
        saved_flyingPercent_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'flyingPercent.npy'
        save_overThresh_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'overThresh.npy'

        # check if the notes have been saved previously, then load
        if ((importAnew==0) & (os.path.isfile(saved_peaksAll_fn)==True)):
            # load previously saved active movement data
            peaksAll = np.load(saved_peaksAll_fn)
            allTraces = np.load(saved_allTraces_fn)
            allDirs = np.load(saved_allDirs_fn)
            allVels = np.load(saved_allVels_fn)
            isFlying = np.load(saved_isFlying_fn)
            flyingPercent = np.load(saved_flyingPercent_fn)
            overThresh = np.load(save_overThresh_fn)
        else: # load notes from scratch (first time analyzing or re-analyzing)
            print('Importing active movements for this experiment: '+flyExpt)
            # import active movement
            isFlying, flyingPercent, overThresh = getFlightsSingleExpt(flyExpt, cameraView, importAnew)
            peaksAll, allTraces, allDirs, allVels = get_active_movements(flyExpt, cameraView,allowFlight=1,TEST=0,importAnew=0);
            isFlying, flyingPercent, overThresh = getFlightsSingleExpt(flyExpt, cameraView, importAnew=0)

            # save active movements
            np.save(saved_peaksAll_fn, peaksAll)
            np.save(saved_allTraces_fn, allTraces)
            np.save(saved_allDirs_fn, allDirs)
            np.save(saved_allVels_fn, allVels)
            np.save(saved_isFlying_fn, isFlying)
            np.save(saved_flyingPercent_fn, flyingPercent)
            np.save(save_overThresh_fn, overThresh)

        # determine directions of wind we're plotting
        valveStates = exptNotes.valveState[:]
        # determine how many speeds of wind were presented and should be plotted
        windVel = exptNotes.velocity[:]
        windVel = windVel.str.get(0) #take values out of brackets
        speeds = np.unique(windVel)

        if secondSeg == 1:
            antIdxs = [0,2] # look at third segments of right and left antenna (can also compare [0,2] to look at second segment)
        else:
            antIdxs = [1,3] # look at third segments of right and left antenna (can also compare [0,2] to look at second segment)

        if noteInd == 0: # set up the figure and some variables at start of iteration
            numVels = np.shape(speeds)[0]
            fig_nonflying, axRast_nonflying = plt.subplots(numVels,facecolor=const.figColor,figsize=(7,10)) #3 columns for nonflying, flight, and partial flight trials
            fig_flying, axRast_flying = plt.subplots(numVels,facecolor=const.figColor,figsize=(7,10)) #3 columns for nonflying, flight, and partial flight trials
            fig_someFlight, axRast_someFlight = plt.subplots(numVels,facecolor=const.figColor,figsize=(7,10)) #3 columns for nonflying, flight, and partial flight trials
            fig_allTrials, axRast_allTrials = plt.subplots(numVels,facecolor=const.figColor,figsize=(7,10))

            numTrials_nonflying = np.zeros([numVels], int) # [0,0,0,0,0] # a tally of all the traces (for spacing in plot)
            numTrials_flying = np.zeros([numVels], int) # a tally of all the traces (for spacing in plot)
            numTrials_someFlight = np.zeros([numVels], int) # a tally of all the traces (for spacing in plot)
            numTrials_allTrials = np.zeros([numVels], int) #this is the worst variable name, whoops...

            # set up a set of arrays to tally up active movements
            pkTally_nonflying = np.zeros([numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)
            pkTally_flying = np.zeros([numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)
            pkTally_someFlight = np.zeros([numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)
            pkTally_allTrials = np.zeros([numVels, 480], int)
            pkTally_flightPerMovement = np.zeros([numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)

            numTrials_nonflying_allFlies = np.zeros([np.shape(allExpts)[0], numVels], int) # [0,0,0,0,0] # a tally of all the traces (for spacing in plot)
            numTrials_flying_allFlies = np.zeros([np.shape(allExpts)[0], numVels], int) # a tally of all the traces (for spacing in plot)
            numTrials_someFlight_allFlies = np.zeros([np.shape(allExpts)[0], numVels], int) # a tally of all the traces (for spacing in plot)
            numTrials_allTrials_allFlies = np.zeros([np.shape(allExpts)[0], numVels], int)

            pkTally_nonflying_allFlies = np.zeros([np.shape(allExpts)[0], numVels, 480], int)
            pkTally_flying_allFlies = np.zeros([np.shape(allExpts)[0], numVels, 480], int)
            pkTally_someFlight_allFlies = np.zeros([np.shape(allExpts)[0], numVels, 480], int)
            pkTally_allTrials_allFlies = np.zeros([np.shape(allExpts)[0], numVels, 480], int)
            pkTally_flightPerMovement_allFlies = np.zeros([np.shape(allExpts)[0], numVels, 480], int) # LxNxM array, L=nonflying/flying/partialflight, N = number of conditions (e.g. ipsi wind) and M = #frames in video (e.g. 480)

            numTrials = np.zeros([numVels],int) # a tally of how many trials have active movements

        # iterate through active movement time points and plot as a raster
        for antInd, ant in enumerate(antIdxs):
            for trialNum in range(np.shape(exptNotes)[0]):
                vel = allVels[trialNum]  # grab the speed, plot accordingly!
                vel_ind = np.squeeze(np.where(vel==speeds))

                if (plotInactivate==0) | ((plotInactivate==1) & activate[trialNum]) | ((plotInactivate==2) & ~activate[trialNum]):
                    peaks = peaksAll[trialNum,ant,:]

                    if peaks.size != 0:
                        if sum(peaks) > 0:
                            numTrials[vel_ind] += 1
                        pks = np.where(peaksAll[trialNum,ant,:]>0)
                        if isFlying[trialNum] == 1:
                            peaksY = np.ones(np.shape(pks))*(-2)-numTrials_flying[vel_ind]*ysft_raster
                            axRast_flying[vel_ind].plot(pks, peaksY,'.',color=plotColor,linewidth=const.traceWidRaw,markersize=const.rasterMkrSizeAllDirs)
                            numTrials_flying[vel_ind] += 1
                            numTrials_flying_allFlies[noteInd, vel_ind] += 1
                        elif isFlying[trialNum]:
                            peaksY = np.ones(np.shape(pks))*(-2)-numTrials_someFlight[vel_ind]*ysft_raster
                            axRast_someFlight[vel_ind].plot(pks, peaksY,'.',color=plotColor,linewidth=const.traceWidRaw,markersize=const.rasterMkrSizeAllDirs)
                            numTrials_someFlight[vel_ind] += 1
                            numTrials_someFlight_allFlies[noteInd, vel_ind] += 1
                        else: #nonflying
                            peaksY = np.ones(np.shape(pks))*(-2)-numTrials_nonflying[vel_ind]*ysft_raster
                            axRast_nonflying[vel_ind].plot(pks, peaksY,'.',color=plotColor,linewidth=const.traceWidRaw,markersize=const.rasterMkrSizeAllDirs)
                            numTrials_nonflying[vel_ind] += 1
                            numTrials_nonflying_allFlies[noteInd,vel_ind] += 1

                        # cross-condition raster for 200cm/s
                        peaksY = np.ones(np.shape(pks))*(-2)-numTrials_allTrials[vel_ind]*ysft_raster
                        axRast_allTrials[vel_ind].plot(pks, peaksY,'.',color=plotColor,linewidth=const.traceWidRaw,markersize=const.rasterMkrSizeAllDirs)
                        numTrials_allTrials[vel_ind] += 1
                        numTrials_allTrials_allFlies[noteInd,vel_ind] += 1

                        for ii,pk in enumerate(pks):
                            if isFlying[trialNum] == 1:
                                pkTally_flying[vel_ind,pk] += 1
                                pkTally_flying_allFlies[noteInd,vel_ind,pk] += 1 #tally up for each fly
                            elif isFlying[trialNum]:
                                pkTally_someFlight[vel_ind,pk] += 1
                                pkTally_someFlight_allFlies[noteInd,vel_ind,pk] += 1 #tally up for each fly
                            else:
                                pkTally_nonflying[vel_ind,pk] += 1
                                pkTally_nonflying_allFlies[noteInd,vel_ind,pk] += 1 #tally up for each fly
                            pkTally_allTrials[vel_ind,pk] += 1
                            pkTally_allTrials_allFlies[noteInd,vel_ind,pk] += 1 #tally up for each fly
                            # additional count = for *each* movement: is it during flight or not
                            # (above tallies classify whole trials as fully or partially flying)
                            if overThresh[trialNum,pk].any():
                                pkTally_flightPerMovement[vel_ind,pk] += 1
                                pkTally_flightPerMovement_allFlies[noteInd,vel_ind,pk] += 1

    # Plot histogram beneath raster, and configure the subplots at the same time
    bb, aa = generate_butter_filter(const.framerate,fc=5) #filter for spike histogram
    if plotInactivate == 0:
        fig_nonflying.suptitle(expt+' nonflying trials',color=const.axisColor, horizontalalignment='center')
        fig_flying.suptitle(expt+' flying trials',color=const.axisColor, horizontalalignment='center')
        fig_someFlight.suptitle(expt+' partially flying trials',color=const.axisColor, horizontalalignment='center')
        fig_allTrials.suptitle(expt+' all trials',color=const.axisColor, horizontalalignment='center')
    else:
        if plotInactivate == 1: actText = ' light (Gtacr inactivation)'
        else: actText = ' no light (no Gtacr inactivation)'
        fig_nonflying.suptitle(expt+' nonflying trials ' + actText,color=const.axisColor, horizontalalignment='center')
        fig_flying.suptitle(expt+' flying trials ' + actText,color=const.axisColor, horizontalalignment='center')
        fig_someFlight.suptitle(expt+' partially flying trials ' + actText,color=const.axisColor, horizontalalignment='center')
        fig_allTrials.suptitle(expt+' all trials ' + actText,color=const.axisColor, horizontalalignment='center')
    scaleHist = 100
    histShiftY = 12
    scaleWidth = 2
    scaleBarSize = 3 # absolute count?
    scaleX = const.numFrames # x-position of scale bar (movements/sec)
    scaleY = -20
    maxAllCt = -np.max([numTrials_nonflying, numTrials_flying, numTrials_someFlight])
    ylimAll = [maxAllCt*ysft_raster-80, 5]
    ylimAll_allTrials = [maxAllCt*ysft_raster-450, 5] #these shifts should really be fully variables based on the #of trials...
    xlimAll = [0, const.numFrames+50]
    # light activation stimulus bar params
    ymin_nonflying = -np.max(numTrials_nonflying)*ysft_raster-histShiftY-4
    ymin_flying = -np.max(numTrials_flying)*ysft_raster-histShiftY-4
    ymin_someFlight = -np.max(numTrials_someFlight)*ysft_raster-histShiftY-4
    rectHeight = const.stimBar_height_allDir
    rectHeight_allTrials = rectHeight*3

    # wind stimulus bar params
    rectX_wind = int(const.windStart*const.framerate)-1  # -1 to align with 0 indexing
    rectY_wind_base = rectHeight*5
    rectWid_wind = int(const.windTime*const.framerate)

    rectX_light = int(const.activateStart*const.framerate)-1  # -1 to align with 0 indexing
    rectY_light_base = rectY_wind_base-rectHeight*4  # add some more to accomodate wind stimulus (above)
    rectWid_light = int(const.activateTime*const.framerate)

    for vel_ind, vel in enumerate(speeds):
        # for nonflying
        tally = (pkTally_nonflying[vel_ind,:]/numTrials_nonflying[vel_ind])*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
        psth_nonflying = signal.filtfilt(bb,aa,tally) # low-pass signals and scale        #axRast_nonflying[vel_ind].plot(np.arange(0,480), angFilt-np.max(numTrials_nonflying)*ysft_raster-histShiftY,linestyle='-',color=const.axisColor,alpha=const.transparencyHist) #plot filtered spike count
        axRast_nonflying[vel_ind].plot(np.arange(0,480), psth_nonflying*scaleHist+ylimAll[0]+scaleBarSize*10,linestyle='-',color=const.axisColor,alpha=const.transparencyHist) #plot filtered spike count
        # configure the subplots here too
        axRast_nonflying[vel_ind].set_facecolor(const.figColor)
        axRast_nonflying[vel_ind].set_ylim(ylimAll)
        axRast_nonflying[vel_ind].set_xlim(xlimAll)
        axRast_nonflying[vel_ind].axis('off')

        # configure and plot histogram for flight
        tally = (pkTally_flying[vel_ind,:]/numTrials_flying[vel_ind])*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
        psth_flying = signal.filtfilt(bb,aa,tally) # low-pass signals and scale
        axRast_flying[vel_ind].plot(np.arange(0,480), psth_flying*scaleHist+ylimAll[0]+scaleBarSize*10,linestyle='-',color=const.axisColor,alpha=const.transparencyHist) #plot filtered spike count
        # configure the subplots here too
        axRast_flying[vel_ind].set_facecolor(const.figColor)
        axRast_flying[vel_ind].axis('off')
        axRast_flying[vel_ind].set_ylim(ylimAll)
        axRast_flying[vel_ind].set_xlim(xlimAll)

        # configure and plot histogram for flight
        tally = (pkTally_someFlight[vel_ind,:]/numTrials_someFlight[vel_ind])*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
        psth_someFlight = signal.filtfilt(bb,aa,tally) # low-pass signals and scale
        axRast_someFlight[vel_ind].plot(np.arange(0,480), psth_someFlight*scaleHist+ylimAll[0]+scaleBarSize*10,linestyle='-',color=const.axisColor,alpha=const.transparencyHist) #plot filtered spike count
        # configure the subplots here too
        axRast_someFlight[vel_ind].set_facecolor(const.figColor)
        axRast_someFlight[vel_ind].axis('off')
        axRast_someFlight[vel_ind].set_ylim(ylimAll)
        axRast_someFlight[vel_ind].set_xlim(xlimAll)

        # configure and plot histogram for all trials
        tally = (pkTally_allTrials[vel_ind,:]/numTrials_allTrials[vel_ind])*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
        psth_allTrials = signal.filtfilt(bb,aa,tally) # low-pass signals and scale
        axRast_allTrials[vel_ind].plot(np.arange(0,480), psth_allTrials*scaleHist+ylimAll_allTrials[0]+scaleBarSize*10,linestyle='-',color=const.axisColor,alpha=const.transparencyHist) #plot filtered spike count
        # configure the subplots here too
        axRast_allTrials[vel_ind].set_facecolor(const.figColor)
        axRast_allTrials[vel_ind].axis('off')
        axRast_allTrials[vel_ind].set_ylim(ylimAll_allTrials)
        axRast_allTrials[vel_ind].set_xlim(xlimAll)

        axRast_nonflying[vel_ind].set_title(str(vel)+' cm/s',color=const.axisColor, horizontalalignment='center',fontsize=const.fsize_raster)
        axRast_flying[vel_ind].set_title(str(vel)+' cm/s',color=const.axisColor, horizontalalignment='left',fontsize=const.fsize_raster)
        axRast_someFlight[vel_ind].set_title(str(vel)+' cm/s',color=const.axisColor, horizontalalignment='right',fontsize=const.fsize_raster)
        axRast_allTrials[vel_ind].set_title(str(vel)+' cm/s',color=const.axisColor, horizontalalignment='left',fontsize=const.fsize_raster)

        # plot movements/sec scale bar
        axRast_nonflying[vel_ind].add_patch(Rectangle((scaleX,-np.max(numTrials_nonflying)*ysft_raster-histShiftY),scaleWidth,scaleBarSize*scaleHist,
            facecolor = const.axisColor))
        axRast_flying[vel_ind].add_patch(Rectangle((scaleX,-np.max(numTrials_flying)*ysft_raster-histShiftY),scaleWidth,scaleBarSize*scaleHist,
            facecolor = const.axisColor))
        axRast_someFlight[vel_ind].add_patch(Rectangle((scaleX,-np.max(numTrials_someFlight)*ysft_raster-histShiftY),scaleWidth,scaleBarSize*scaleHist,
            facecolor = const.axisColor))
        axRast_allTrials[vel_ind].add_patch(Rectangle((scaleX,-np.max(numTrials_allTrials)*ysft_raster-histShiftY),scaleWidth,scaleBarSize*scaleHist,
            facecolor = const.axisColor))
        #label scale bar
        axRast_nonflying[vel_ind].text(scaleX+scaleWidth*5,-np.max(numTrials_nonflying)*ysft_raster-histShiftY,str(scaleBarSize)+' Hz',fontsize=const.fontSize_stimBar,color=const.axisColor)
        axRast_flying[vel_ind].text(scaleX+scaleWidth*5,-np.max(numTrials_flying)*ysft_raster-histShiftY,str(scaleBarSize)+' Hz',fontsize=const.fontSize_stimBar,color=const.axisColor)
        axRast_someFlight[vel_ind].text(scaleX+scaleWidth*5,-np.max(numTrials_someFlight)*ysft_raster-histShiftY,str(scaleBarSize)+' Hz',fontsize=const.fontSize_stimBar,color=const.axisColor)
        axRast_allTrials[vel_ind].text(scaleX+scaleWidth*5,-np.max(numTrials_allTrials)*ysft_raster-histShiftY,str(scaleBarSize)+' Hz',fontsize=const.fontSize_stimBar,color=const.axisColor)

        # plot the wind/light stimulus regions below the data
        axRast_nonflying[vel_ind].add_patch(Rectangle((rectX_wind, ylimAll[0]+rectY_wind_base), rectWid_wind, rectHeight,
            facecolor=const.axisColor))
        axRast_nonflying[vel_ind].text(rectX_wind+rectWid_wind+5, ylimAll[0]+rectY_wind_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
            str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='left',
            color=const.axisColor)
        axRast_flying[vel_ind].add_patch(Rectangle((rectX_wind, ylimAll[0]+rectY_wind_base), rectWid_wind, rectHeight,
            facecolor=const.axisColor))
        axRast_flying[vel_ind].text(rectX_wind+rectWid_wind+5, ylimAll[0]+rectY_wind_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
            str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='left',
            color=const.axisColor)
        axRast_someFlight[vel_ind].add_patch(Rectangle((rectX_wind, ylimAll[0]+rectY_wind_base), rectWid_wind, rectHeight,
            facecolor=const.axisColor))
        axRast_someFlight[vel_ind].text(rectX_wind+rectWid_wind+5, ylimAll[0]+rectY_wind_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
            str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='left',
            color=const.axisColor)

        axRast_allTrials[vel_ind].add_patch(Rectangle((rectX_wind, ylimAll_allTrials[0]+rectY_wind_base), rectWid_wind, rectHeight_allTrials,
            facecolor=const.axisColor))
        axRast_allTrials[vel_ind].text(rectX_wind+rectWid_wind+5, ylimAll_allTrials[0]+rectY_wind_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
            str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='left',
            color=const.axisColor)

        if (plotInactivate != 2):
            axRast_nonflying[vel_ind].add_patch(Rectangle((rectX_light, ylimAll[0]+rectY_light_base), rectWid_light, rectHeight,
                facecolor=const.color_activateColor))
            axRast_nonflying[vel_ind].text(rectX_light+rectWid_light+5, ylimAll[0]+rectY_light_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                str(const.activateTime)+' s light',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                color=const.color_activateColor)
            axRast_flying[vel_ind].add_patch(Rectangle((rectX_light, ylimAll[0]+rectY_light_base), rectWid_light, rectHeight,
                facecolor=const.color_activateColor))
            axRast_flying[vel_ind].text(rectX_light+rectWid_light+5, ylimAll[0]+rectY_light_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                str(const.activateTime)+' s light',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                color=const.color_activateColor)
            axRast_someFlight[vel_ind].add_patch(Rectangle((rectX_light, ylimAll[0]+rectY_light_base), rectWid_light, rectHeight,
                facecolor=const.color_activateColor))
            axRast_someFlight[vel_ind].text(rectX_light+rectWid_light+5, ylimAll[0]+rectY_light_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                str(const.activateTime)+' s light',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                color=const.color_activateColor)
            axRast_allTrials[vel_ind].add_patch(Rectangle((rectX_light, ylimAll_allTrials[0]+rectY_light_base), rectWid_light, rectHeight_allTrials,
                facecolor=const.color_activateColor))
            axRast_allTrials[vel_ind].text(rectX_light+rectWid_light+5, ylimAll_allTrials[0]+rectY_light_base-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
                str(const.activateTime)+' s light',fontsize=const.fontSize_stimBar,horizontalalignment='left',
                color=const.color_activateColor)

    plt.show()
    plt.pause(0.001)
    plt.show(block=False)

    if plotInactivate == 1:
        extraText = '_withLight_'
    elif plotInactivate == 2:
        extraText = '_noLight_'
    else:
        extraText = ''

    saved_pkTally_nonflying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'pkTally_nonflying.npy'
    saved_pkTally_flying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'pkTally_flying.npy'
    saved_pkTally_someFlight_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'pkTally_someFlight.npy'
    saved_pkTally_flightPerMovement_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'pkTally_flightPerMovement.npy'

    saved_pkTally_nonflying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'pkTally_nonflying_allFlies.npy'
    saved_pkTally_flying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'pkTally_flying_allFlies.npy'
    saved_pkTally_someFlight_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'pkTally_someFlight_allFlies.npy'
    saved_pkTally_flightPerMovement_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'pkTally_flightPerMovement_allFlies.npy'

    saved_pkTally_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'pkTally_allTrials_allFlies.npy'
    numTrials_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'numTrials_allTrials_allFlies.npy'

    numTrials_nonflying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'numTrials_nonflying.npy'
    numTrials_flying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'numTrials_flying.npy'
    numTrials_someFlight_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'numTrials_someFlight.npy'

    numTrials_nonflying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'numTrials_nonflying_allFlies.npy'
    numTrials_flying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'numTrials_flying_allFlies.npy'
    numTrials_someFlight_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'numTrials_someFlight_allFlies.npy'

    np.save(saved_pkTally_nonflying_fn, pkTally_nonflying)
    np.save(saved_pkTally_flying_fn, pkTally_flying)
    np.save(saved_pkTally_someFlight_fn, pkTally_someFlight)
    np.save(saved_pkTally_flightPerMovement_fn, pkTally_flightPerMovement)

    np.save(saved_pkTally_nonflying_allFlies_fn, pkTally_nonflying_allFlies)
    np.save(saved_pkTally_flying_allFlies_fn, pkTally_flying_allFlies)
    np.save(saved_pkTally_someFlight_allFlies_fn, pkTally_someFlight_allFlies)
    np.save(saved_pkTally_flightPerMovement_allFlies_fn, pkTally_flightPerMovement_allFlies)

    np.save(saved_pkTally_allTrials_allFlies_fn, pkTally_allTrials_allFlies)
    np.save(numTrials_allTrials_allFlies_fn, numTrials_allTrials_allFlies)


    np.save(numTrials_nonflying_fn, numTrials_nonflying)
    np.save(numTrials_flying_fn, numTrials_flying)
    np.save(numTrials_someFlight_fn, numTrials_someFlight)

    np.save(numTrials_nonflying_allFlies_fn, numTrials_nonflying_allFlies)
    np.save(numTrials_flying_allFlies_fn, numTrials_flying_allFlies)
    np.save(numTrials_someFlight_allFlies_fn, numTrials_someFlight_allFlies)

    #save the histograms for compilation for the figure
    psth_nonflying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_nonflying.npy'
    np.save(psth_nonflying_fn, psth_nonflying)

    psth_flying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_flying.npy'
    np.save(psth_flying_fn, psth_flying)

    psth_someFlight_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_someFlight.npy'
    np.save(psth_someFlight_fn, psth_someFlight)

    psth_allTrials_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_allTrials.npy'
    np.save(psth_allTrials_fn, psth_allTrials)

    if savefig: # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        # load data for active movements, flight
        if plotInactivate == 1:
            extraText = '_withLight_'
        elif plotInactivate == 2:
            extraText = '_noLight_'
        else:
            extraText = ''
        # save raster (cross-fly)
        savepath = figPath + expt + '_raster_alldir_velocity_nonflying' + extraText+'.png'
        fig_nonflying.savefig(savepath, facecolor=fig_nonflying.get_facecolor())
        savepath = figPath + expt + '_raster_alldir_velocity_nonflying' +  extraText+'.pdf'
        fig_nonflying.savefig(savepath, facecolor=fig_nonflying.get_facecolor())

        savepath = figPath + expt + '_raster_alldir_velocity_flying' +  extraText+'.png'
        fig_flying.savefig(savepath, facecolor=fig_flying.get_facecolor())
        savepath = figPath + expt + '_raster_alldir_velocity_flying' +  extraText+'.pdf'
        fig_flying.savefig(savepath, facecolor=fig_flying.get_facecolor())

        savepath = figPath + expt + '_raster_alldir_velocity_partialFlights' +  extraText+'.png'
        fig_someFlight.savefig(savepath, facecolor=fig_someFlight.get_facecolor())
        savepath = figPath + expt + '_raster_alldir_velocity_partialFlights' +  extraText+'.pdf'
        fig_someFlight.savefig(savepath, facecolor=fig_someFlight.get_facecolor())

        savepath = figPath + expt + '_raster_alldir_velocity_allTrials' +  extraText+'.png'
        fig_allTrials.savefig(savepath, facecolor=fig_allTrials.get_facecolor())
        savepath = figPath + expt + '_raster_alldir_velocity_allTrials' +  extraText+'.pdf'
        fig_allTrials.savefig(savepath, facecolor=fig_allTrials.get_facecolor())

# plot psth for inactivation data altogether!
# Relies on previously saved data, i.e. by running the following for each genotype:
#   pt.plot_active_movement_raster_cross_fly_all_dir(expt='91F02_inactivate',cameraView='frontal', plotInactivate=1,importAnew = 0,savefig = 0)
def plot_psth_inactivation(savefig=0):
    cameraView = 'frontal'
    plot_single_fly_avgs = 0
    st = 61 #zoom in on start of light/wind
    stop =228 #zoom in on start of light/wind
    vel_ind = 1
    bb, aa = generate_butter_filter(const.framerate,fc=5) #filter for spike histogram

    # Grab previously saved data (from running above function,, for plotInactivate=1 and 2)
    expt='18D07_inactivate'
    extraText = '_noLight_'
    psth_allTrials_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_allTrials.npy'
    psth_allTrials_noLight_18D07 = np.load(psth_allTrials_fn)
    saved_pkTally_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'pkTally_allTrials_allFlies.npy'
    pks_18D07_noLight = np.load(saved_pkTally_allTrials_allFlies_fn)
    numTrials_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'numTrials_allTrials_allFlies.npy'
    numTrials_18D07_noLight = np.load(numTrials_allTrials_allFlies_fn)
    extraText = '_withLight_'
    psth_allTrials_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_allTrials.npy'
    psth_allTrials_light_18D07 = np.load(psth_allTrials_fn)
    saved_pkTally_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'pkTally_allTrials_allFlies.npy'
    pks_18D07_withLight = np.load(saved_pkTally_allTrials_allFlies_fn)
    numTrials_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'numTrials_allTrials_allFlies.npy'
    numTrials_18D07_withLight = np.load(numTrials_allTrials_allFlies_fn)

    expt='91F02_inactivate'
    extraText = '_noLight_'
    psth_allTrials_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_allTrials.npy'
    psth_allTrials_noLight_91F02 = np.load(psth_allTrials_fn)
    saved_pkTally_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'pkTally_allTrials_allFlies.npy'
    pks_91F02_noLight = np.load(saved_pkTally_allTrials_allFlies_fn)
    numTrials_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'numTrials_allTrials_allFlies.npy'
    numTrials_91F02_noLight = np.load(numTrials_allTrials_allFlies_fn)
    extraText = '_withLight_'
    psth_allTrials_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_allTrials.npy'
    psth_allTrials_light_91F02 = np.load(psth_allTrials_fn)
    saved_pkTally_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'pkTally_allTrials_allFlies.npy'
    pks_91F02_withLight = np.load(saved_pkTally_allTrials_allFlies_fn)
    numTrials_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'numTrials_allTrials_allFlies.npy'
    numTrials_91F02_withLight = np.load(numTrials_allTrials_allFlies_fn)

    expt='emptyGAL4_inactivate'
    extraText = '_noLight_'
    psth_allTrials_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_allTrials.npy'
    psth_allTrials_noLight_emptyGAL4 = np.load(psth_allTrials_fn)
    saved_pkTally_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'pkTally_allTrials_allFlies.npy'
    pks_emptyGAL4_noLight = np.load(saved_pkTally_allTrials_allFlies_fn)
    numTrials_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'numTrials_allTrials_allFlies.npy'
    numTrials_emptyGAL4_noLight = np.load(numTrials_allTrials_allFlies_fn)
    extraText = '_withLight_'
    psth_allTrials_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+'psth_allTrials.npy'
    psth_allTrials_light_emptyGAL4 = np.load(psth_allTrials_fn)
    saved_pkTally_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'pkTally_allTrials_allFlies.npy'
    pks_emptyGAL4_withLight = np.load(saved_pkTally_allTrials_allFlies_fn)
    numTrials_allTrials_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+extraText+'numTrials_allTrials_allFlies.npy'
    numTrials_emptyGAL4_withLight = np.load(numTrials_allTrials_allFlies_fn)

    fig_allTrials, axRast_allTrials = plt.subplots(1,3,facecolor=const.figColor,figsize=(14,5))

    # plot single fly data
    jack_std_emptyGAL4_noLight = get_jacknife_std(pks_emptyGAL4_noLight[:,vel_ind,:],numTrials_emptyGAL4_noLight[:,vel_ind],psth_allTrials_noLight_emptyGAL4,bb,aa)
    jack_std_emptyGAL4_withLight = get_jacknife_std(pks_emptyGAL4_withLight[:,vel_ind,:],numTrials_emptyGAL4_withLight[:,vel_ind],psth_allTrials_light_emptyGAL4,bb,aa)
    jack_std_18D07_noLight = get_jacknife_std(pks_18D07_noLight[:,vel_ind,:],numTrials_18D07_noLight[:,vel_ind],psth_allTrials_noLight_18D07,bb,aa)
    jack_std_18D07_withLight = get_jacknife_std(pks_18D07_withLight[:,vel_ind,:],numTrials_18D07_withLight[:,vel_ind],psth_allTrials_light_18D07,bb,aa)
    jack_std_91F02_noLight = get_jacknife_std(pks_91F02_noLight[:,vel_ind,:],numTrials_91F02_noLight[:,vel_ind],psth_allTrials_noLight_91F02,bb,aa)
    jack_std_91F02_withLight = get_jacknife_std(pks_91F02_withLight[:,vel_ind,:],numTrials_91F02_withLight[:,vel_ind],psth_allTrials_light_91F02,bb,aa)

    #plot cross-fly data
    xx = np.linspace(0,stop-st,stop-st)
    #emptyGAL4 (control)
    axRast_allTrials[0].plot(psth_allTrials_noLight_emptyGAL4[st:stop], color=const.color_inactivationNoLight)
    axRast_allTrials[0].fill_between(xx,psth_allTrials_noLight_emptyGAL4[st:stop]-jack_std_emptyGAL4_noLight[st:stop],
        psth_allTrials_noLight_emptyGAL4[st:stop]+jack_std_emptyGAL4_noLight[st:stop], edgecolor=None,facecolor=const.color_inactivationNoLight,alpha=const.transparencyPatch)
    axRast_allTrials[0].plot(psth_allTrials_light_emptyGAL4[st:stop], color=const.color_inactivationLight)
    axRast_allTrials[0].fill_between(xx,psth_allTrials_light_emptyGAL4[st:stop]-jack_std_emptyGAL4_withLight[st:stop],
        psth_allTrials_light_emptyGAL4[st:stop]+jack_std_emptyGAL4_withLight[st:stop], edgecolor=None,facecolor=const.color_inactivationLight,alpha=const.transparencyPatch)
    #18D07
    axRast_allTrials[1].plot(psth_allTrials_noLight_18D07[st:stop], color=const.color_inactivationNoLight)
    axRast_allTrials[1].fill_between(xx,psth_allTrials_noLight_18D07[st:stop]-jack_std_18D07_noLight[st:stop],
        psth_allTrials_noLight_18D07[st:stop]+jack_std_18D07_noLight[st:stop], edgecolor=None,facecolor=const.color_inactivationNoLight,alpha=const.transparencyPatch)
    axRast_allTrials[1].plot(psth_allTrials_light_18D07[st:stop], color=const.color_inactivationLight)
    axRast_allTrials[1].fill_between(xx,psth_allTrials_light_18D07[st:stop]-jack_std_18D07_withLight[st:stop],
        psth_allTrials_light_18D07[st:stop]+jack_std_18D07_withLight[st:stop], edgecolor=None,facecolor=const.color_inactivationLight,alpha=const.transparencyPatch)
    #91F02
    axRast_allTrials[2].plot(psth_allTrials_noLight_91F02[st:stop], color=const.color_inactivationNoLight)
    axRast_allTrials[2].fill_between(xx,psth_allTrials_noLight_91F02[st:stop]-jack_std_91F02_noLight[st:stop],
        psth_allTrials_noLight_91F02[st:stop]+jack_std_91F02_noLight[st:stop], edgecolor=None,facecolor=const.color_inactivationNoLight,alpha=const.transparencyPatch)
    axRast_allTrials[2].plot(psth_allTrials_light_91F02[st:stop], color=const.color_inactivationLight)
    axRast_allTrials[2].fill_between(xx,psth_allTrials_light_91F02[st:stop]-jack_std_91F02_withLight[st:stop],
        psth_allTrials_light_91F02[st:stop]+jack_std_91F02_withLight[st:stop], edgecolor=None,facecolor=const.color_inactivationLight,alpha=const.transparencyPatch)

    #Configure the subplots
    titles = ['PBD>Gtacr','18D07>Gtacr','91F02>Gtacr']
    # wind stimulus bar params
    ylimAll = [-0.2, 2.7]
    scaleX = 168
    scaleY = 0
    scaleWidth = 2
    scaleBarSize = 0.5
    rectHeight = 0.05
    rectX_wind = int(const.windStart*const.framerate)-1-st  # -1 to align with 0 indexing
    rectY_wind_base = rectHeight*2
    rectWid_wind = int(const.windTime*const.framerate)-(int(const.windStop*const.framerate)-stop)
    timeWind = rectWid_wind*1000/const.framerate #convert to ms

    rectX_light = int(const.activateStart*const.framerate)-1-st  # -1 to align with 0 indexing
    rectY_light_base = rectY_wind_base-rectHeight*2  # add some more to accomodate wind stimulus (above)
    rectWid_light = int(const.activateTime*const.framerate)-(int(const.activateStop*const.framerate)-stop)
    timeLight = rectWid_light*1000/const.framerate #convert to ms


    for ii in [0,1,2]:
        axRast_allTrials[ii].set_ylim(ylimAll)
        axRast_allTrials[ii].axis('off')
        axRast_allTrials[ii].add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
            facecolor = const.axisColor))
        #label scale bar
        axRast_allTrials[ii].text(scaleX+scaleWidth*2,scaleY+scaleBarSize/2,str(scaleBarSize)+' Hz',fontsize=const.fontSize_stimBar,color=const.axisColor)

        axRast_allTrials[ii].add_patch(Rectangle((rectX_wind, ylimAll[0]+rectY_wind_base), rectWid_wind, rectHeight,
            facecolor=const.axisColor))
        axRast_allTrials[ii].text(rectX_wind+rectWid_wind+5, ylimAll[0]+rectY_wind_base-rectHeight,
            str(timeWind)+' ms wind',fontsize=const.fontSize_stimBar,horizontalalignment='left',
            color=const.axisColor)

        axRast_allTrials[ii].add_patch(Rectangle((rectX_light, ylimAll[0]+rectY_light_base), rectWid_light, rectHeight,
            facecolor=const.color_inactivateColor))
        axRast_allTrials[ii].text(rectX_light+rectWid_light+5, ylimAll[0]+rectY_light_base-rectHeight,
            str(timeLight)+' ms light',fontsize=const.fontSize_stimBar,horizontalalignment='left',
            color=const.color_inactivateColor)

        axRast_allTrials[ii].set_title(titles[ii])

    plt.show()
    plt.pause(0.001)
    plt.show(block=False)

    if savefig:
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        # save raster (cross-fly)
        savepath = figPath + 'inactivation_raster_overlay' + extraText+'.png'
        fig_allTrials.savefig(savepath, facecolor=fig_allTrials.get_facecolor())
        savepath = figPath + 'inactivation_raster_overlay' +  extraText+'.pdf'
        fig_allTrials.savefig(savepath, facecolor=fig_allTrials.get_facecolor())

# Get jacknife variance (quite hard-coded for plot_psth_inactivation function above)
def get_jacknife_std(pks,numTrials,psth_allTrials,bb,aa):

    sumDiff = 0
    numFlies = np.shape(pks)[0]
    for flyNum in range(numFlies):
        tally = (pks[flyNum,:]/numTrials[flyNum])*const.framerate #normalize to number of trials and multiply by framerate to get movements/sec
        psth_oneFly = signal.filtfilt(bb,aa,tally)
        sumDiff = sumDiff+(psth_oneFly-psth_allTrials)**2

    var_jack_18D07_noLight = (1/(numFlies*(numFlies-1)))*sumDiff
    jack_std_18D07_noLight = np.sqrt(var_jack_18D07_noLight)
    return jack_std_18D07_noLight

# Quantify how many active movements occur during wind (across directions and velocity)
# --> For flight vs. nonflying
# if secondSeg == 0, will default to third segment. If 1, will get second segment data.
def get_movement_count_during_wind(expt='CS_activate',cameraView='frontal',plotInactivate=0,importAnew = 0,savefig = 0,secondSeg=1):
    # get this experiment type list of experiments from notes
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)

    # load data for active movements, flight
    if plotInactivate == 1:
        extraText = 'withLight_'
    elif plotInactivate == 2:
        extraText = 'noLight_'
    else:
        extraText = ''
    if secondSeg == 1:
        segmentText = 'secondSeg_'
    else: segmentText = 'thirdSeg_'

    pkTally_nonflying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_nonflying.npy'
    pkTally_flightPerMovement_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_flightPerMovement.npy'

    pkTally_nonflying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_nonflying_allFlies.npy'
    pkTally_flightPerMovement_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'pkTally_flightPerMovement_allFlies.npy'

    numTrials_nonflying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_nonflying.npy'
    numTrials_someFlight_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_someFlight.npy'
    numTrials_flying_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_flying.npy'
    numTrials_nonflying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_nonflying_allFlies.npy'
    numTrials_someFlight_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_someFlight_allFlies.npy'
    numTrials_flying_allFlies_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+extraText+segmentText+'numTrials_flying_allFlies.npy'

    print(cameraView)
    print(extraText)
    print(pkTally_nonflying_fn)

    if (importAnew==1):
        print('Re-computing active movement counts from ''plot_active_movement_raster_cross_fly''')
        plot_active_movement_raster_cross_fly(expt,cameraView,plotInactivate=plotInactivate,importAnew = 1,savefig = 0,secondSeg=secondSeg);
    elif (os.path.isfile(numTrials_someFlight_allFlies_fn)==False):# | (secondSeg>0):
        print('Importing active movement tallies from ''plot_active_movement_raster_cross_fly''')
        print(secondSeg)
        plot_active_movement_raster_cross_fly(expt,cameraView,plotInactivate=plotInactivate,importAnew = 0,savefig = 0,secondSeg=secondSeg);

    # load saved data
    # also grab number of trials, need to normalize to this! (way fewer flight trials)
    pkTally_nonflying = np.load(pkTally_nonflying_fn)
    pkTally_flightPerMovement = np.load(pkTally_flightPerMovement_fn)
    pkTally_nonflying_allFlies = np.load(pkTally_nonflying_allFlies_fn)
    pkTally_flightPerMovement_allFlies = np.load(pkTally_flightPerMovement_allFlies_fn)

    numTrials_nonflying = np.load(numTrials_nonflying_fn)
    numTrials_someFlight = np.load(numTrials_someFlight_fn)
    numTrials_flying = np.load(numTrials_flying_fn)
    numTrials_all = numTrials_nonflying+numTrials_someFlight+numTrials_flying #tally of all trials
    numTrials_nonflying_allFlies = np.load(numTrials_nonflying_allFlies_fn)
    numTrials_someFlight_allFlies = np.load(numTrials_someFlight_allFlies_fn)
    numTrials_flying_allFlies = np.load(numTrials_flying_allFlies_fn)

    peak_count_nonflying = np.zeros([np.shape(pkTally_flightPerMovement)[0], np.shape(pkTally_flightPerMovement)[1]], int)
    peak_count_flying = np.zeros([np.shape(pkTally_flightPerMovement)[0], np.shape(pkTally_flightPerMovement)[1]], int)

    # compute average number of active movements, normalized by # trials per fly
    flyExpt = allExpts[0]['date']
    exptNotes = get_mat_notes(flyExpt)
    stimStart = exptNotes.pre_trial_time[0][0]
    stimStop = exptNotes.pre_trial_time[0][0]+exptNotes.trial_time_wind[0][0]
    windStart = stimStart*const.framerate
    windStop = stimStop*const.framerate
    windInds = np.arange(windStart, windStop)
    preWindInds = np.arange(0, windStart-1)
    postWindInds = np.arange(windStop, const.numFrames-1)

    # movements during wind
    ct_wind_nonflying = np.sum(pkTally_nonflying[:,:,windInds],axis=2)
    ct_wind_flying = np.sum(pkTally_flightPerMovement[:,:,windInds],axis=2)
    ct_wind_all = np.sum(pkTally_flightPerMovement[:,:,windInds]+pkTally_nonflying[:,:,windInds],axis=2)
    wind_nonflyingNorm = ct_wind_nonflying/numTrials_nonflying
    wind_flyingNorm = ct_wind_flying/(numTrials_someFlight+numTrials_flying)
    wind_allNorm = ct_wind_all/numTrials_all

    ct_wind_nonflying_allFlies = np.sum(pkTally_nonflying_allFlies[:,:,:,windInds],axis=3)
    ct_wind_flying_allFlies = np.sum(pkTally_flightPerMovement_allFlies[:,:,:,windInds],axis=3)
    ct_wind_all_allFlies = np.sum(pkTally_nonflying_allFlies[:,:,:,windInds]+pkTally_flightPerMovement_allFlies[:,:,:,windInds],axis=3)

    wind_nonflyingNorm_allFlies = ct_wind_nonflying_allFlies/numTrials_nonflying_allFlies
    wind_flyingNorm_allFlies = ct_wind_flying_allFlies/(numTrials_someFlight_allFlies+numTrials_flying_allFlies)
    wind_allNorm_allFlies = ct_wind_all_allFlies/(numTrials_nonflying_allFlies+numTrials_someFlight_allFlies+numTrials_flying_allFlies)
    wind_allNotNorm_allFlies = ct_wind_all_allFlies.astype(float)

    wind_nonflyingNorm_allFlies[np.where(np.isinf(wind_nonflyingNorm_allFlies))] = np.nan
    wind_flyingNorm_allFlies[np.where(np.isinf(wind_flyingNorm_allFlies))] = np.nan
    wind_allNorm_allFlies[np.where(np.isinf(wind_allNorm_allFlies))] = np.nan
    wind_allNotNorm_allFlies[np.where(np.isinf(wind_allNotNorm_allFlies))] = np.nan


    # movements before wind
    ct_prewind_nonflying = np.sum(pkTally_nonflying[:,:,preWindInds],axis=2)
    ct_prewind_flying = np.sum(pkTally_flightPerMovement[:,:,preWindInds],axis=2)
    prewind_nonflyingNorm = ct_prewind_nonflying/numTrials_nonflying
    prewind_flyingNorm = ct_prewind_flying/(numTrials_someFlight+numTrials_flying)

    ct_prewind_nonflying_allFlies = np.sum(pkTally_nonflying_allFlies[:,:,:,preWindInds],axis=3)
    ct_prewind_flying_allFlies = np.sum(pkTally_flightPerMovement_allFlies[:,:,:,preWindInds],axis=3)
    prewind_nonflyingNorm_allFlies = ct_prewind_nonflying_allFlies/numTrials_nonflying_allFlies
    prewind_flyingNorm_allFlies = ct_prewind_flying_allFlies/(numTrials_someFlight_allFlies+numTrials_flying_allFlies)
    prewind_nonflyingNorm_allFlies[np.where(np.isinf(prewind_nonflyingNorm_allFlies))] = np.nan
    prewind_flyingNorm_allFlies[np.where(np.isinf(prewind_flyingNorm_allFlies))] = np.nan

    # movements after wind
    ct_postwind_nonflying = np.sum(pkTally_nonflying[:,:,postWindInds],axis=2)
    ct_postwind_flying = np.sum(pkTally_flightPerMovement[:,:,postWindInds],axis=2)
    postwind_nonflyingNorm = ct_postwind_nonflying/numTrials_nonflying
    postwind_flyingNorm = ct_postwind_flying/(numTrials_someFlight+numTrials_flying)

    # movements after wind
    ct_postwind_nonflying_allFlies = np.sum(pkTally_nonflying_allFlies[:,:,:,postWindInds],axis=3)
    ct_postwind_flying_allFlies = np.sum(pkTally_flightPerMovement_allFlies[:,:,:,postWindInds],axis=3)
    postwind_nonflyingNorm_allFlies = ct_postwind_nonflying_allFlies/numTrials_nonflying_allFlies
    postwind_flyingNorm_allFlies = ct_postwind_flying_allFlies/(numTrials_someFlight_allFlies+numTrials_flying_allFlies)
    postwind_nonflyingNorm_allFlies[np.where(np.isinf(postwind_nonflyingNorm_allFlies))] = np.nan
    postwind_flyingNorm_allFlies[np.where(np.isinf(postwind_flyingNorm_allFlies))] = np.nan

    # nan any trials that do not meet minimum trial number threshold
    wind_nonflyingNorm[np.where(numTrials_nonflying<const.trialThreshForQuant)] = np.nan
    wind_flyingNorm[np.where((numTrials_someFlight+numTrials_flying)<const.trialThreshForQuant)] = np.nan
    wind_allNorm[np.where(numTrials_all<const.trialThreshForQuant)] = np.nan

    wind_nonflyingNorm_allFlies[np.where(numTrials_nonflying_allFlies<const.trialThreshForQuant)] = np.nan
    wind_flyingNorm_allFlies[np.where((numTrials_someFlight_allFlies+numTrials_flying_allFlies)<const.trialThreshForQuant)] = np.nan
    wind_allNorm_allFlies[np.where(numTrials_all<const.trialThreshForQuant)] = np.nan
    wind_allNotNorm_allFlies[np.where(numTrials_all<const.trialThreshForQuant)] = np.nan

    prewind_nonflyingNorm[np.where(numTrials_nonflying<const.trialThreshForQuant)] = np.nan
    prewind_flyingNorm[np.where((numTrials_someFlight+numTrials_flying)<const.trialThreshForQuant)] = np.nan

    prewind_nonflyingNorm_allFlies[np.where(numTrials_nonflying_allFlies<const.trialThreshForQuant)] = np.nan
    prewind_flyingNorm_allFlies[np.where((numTrials_someFlight_allFlies+numTrials_flying_allFlies)<const.trialThreshForQuant)] = np.nan

    postwind_nonflyingNorm[np.where(numTrials_nonflying<const.trialThreshForQuant)] = np.nan
    postwind_flyingNorm[np.where((numTrials_someFlight+numTrials_flying)<const.trialThreshForQuant)] = np.nan

    postwind_nonflyingNorm_allFlies[np.where(numTrials_nonflying_allFlies<const.trialThreshForQuant)] = np.nan
    postwind_flyingNorm_allFlies[np.where((numTrials_someFlight_allFlies+numTrials_flying_allFlies)<const.trialThreshForQuant)] = np.nan

    return [prewind_nonflyingNorm_allFlies, prewind_flyingNorm_allFlies, wind_nonflyingNorm_allFlies, wind_flyingNorm_allFlies,
    postwind_nonflyingNorm_allFlies, postwind_flyingNorm_allFlies, wind_allNorm_allFlies, wind_allNotNorm_allFlies]


# use with above function to plot # active movements across all directions and several wind speeds
# used to look at active movement quantification for typical activation experiments (inactivation quantification in 'plot_inactivation_paired_active_movements)
def plot_movement_count_during_wind(expt='CS_activate',cameraView='frontal',importAnew = 0,savefig = 0,secondSeg=1):

    [prewind_nonflyingNorm_allFlies, prewind_flyingNorm_allFlies, wind_nonflyingNorm_allFlies, wind_flyingNorm_allFlies,
    postwind_nonflyingNorm_allFlies, postwind_flyingNorm_allFlies,wind_allNorm_allFlies, wind_allNotNorm_allFlies] = get_movement_count_during_wind(expt,cameraView,0,importAnew,savefig,secondSeg)

    prewind_nonflyingNorm_allFlies[np.where(prewind_nonflyingNorm_allFlies==0)] = np.nan
    prewind_flyingNorm_allFlies[np.where(prewind_flyingNorm_allFlies==0)] = np.nan
    wind_nonflyingNorm_allFlies[np.where(wind_nonflyingNorm_allFlies==0)] = np.nan
    wind_flyingNorm_allFlies[np.where(wind_flyingNorm_allFlies==0)] = np.nan
    wind_allNorm_allFlies[np.where(wind_allNorm_allFlies==0)] = np.nan
    postwind_nonflyingNorm_allFlies[np.where(postwind_nonflyingNorm_allFlies==0)] = np.nan
    postwind_flyingNorm_allFlies[np.where(postwind_flyingNorm_allFlies==0)] = np.nan

    # make the figure
    fig, ax = plt.subplots(3,2,facecolor=const.figColor,figsize=(5,8))
    fig.suptitle('Total number of active movements during wind\nnormalized by # trials',color=const.axisColor)
    ax[0,0].set_title('nonflying trials',color=const.axisColor, horizontalalignment='center',fontsize=const.fsize_raster)
    ax[0,1].set_title('trials with flight',color=const.axisColor, horizontalalignment='center',fontsize=const.fsize_raster)

    fig2, ax2 = plt.subplots(1,1,facecolor=const.figColor,figsize=(5,8))
    fig2.suptitle('Total number of active movements during wind\nnormalized by # trials',color=const.axisColor)

    # plot legend for speeds
    shiftAmt = 0.5
    jitterSft = 0.2
    spInds = [0,1,2,3,4]
    spIndsAll = [spInds]*np.shape(wind_nonflyingNorm_allFlies)[0]
    speeds = [0, 50, 100, 200]
    velsToPlot = [2,3]

    # typically plot curves only for higher speeds, where we have flight data
    for velInd in velsToPlot:  # range(np.shape(wind_nonflyingNorm)[1]):
        jitter = jitterSft*np.random.rand(np.shape(spIndsAll)[0],np.shape(spIndsAll)[1])-jitterSft/2 # add some random jitter to x-placement of single trial responses
        ax[0,0].plot(spInds,np.nanmean(prewind_nonflyingNorm_allFlies[:,:,velInd],axis=0),marker='.',color=const.colors_velocity_RminL[velInd])
        ax[0,0].plot(spIndsAll+jitter,prewind_nonflyingNorm_allFlies[:,:,velInd],marker='.',color=const.colors_velocity_RminL[velInd],linestyle='None')

        jitter = jitterSft*np.random.rand(np.shape(spIndsAll)[0],np.shape(spIndsAll)[1])-jitterSft/2 # add some random jitter to x-placement of single trial responses
        ax[0,1].plot(spInds,np.nanmean(prewind_flyingNorm_allFlies[:,:,velInd],axis=0),marker='.',color=const.colors_velocity_RminL[velInd])
        ax[0,1].plot(spIndsAll+jitter,prewind_flyingNorm_allFlies[:,:,velInd],marker='.',color=const.colors_velocity_RminL[velInd],linestyle='None')

        jitter = jitterSft*np.random.rand(np.shape(spIndsAll)[0],np.shape(spIndsAll)[1])-jitterSft/2 # add some random jitter to x-placement of single trial responses
        ax[1,0].plot(spInds,np.nanmean(wind_nonflyingNorm_allFlies[:,:,velInd],axis=0),marker='.',color=const.colors_velocity_RminL[velInd])
        ax[1,0].plot(spIndsAll+jitter,wind_nonflyingNorm_allFlies[:,:,velInd],marker='.',color=const.colors_velocity_RminL[velInd],linestyle='None')

        jitter = jitterSft*np.random.rand(np.shape(spIndsAll)[0],np.shape(spIndsAll)[1])-jitterSft/2 # add some random jitter to x-placement of single trial responses
        ax[1,1].plot(spInds,np.nanmean(wind_flyingNorm_allFlies[:,:,velInd],axis=0),marker='.',color=const.colors_velocity_RminL[velInd])
        ax[1,1].plot(spIndsAll+jitter,wind_flyingNorm_allFlies[:,:,velInd],marker='.',color=const.colors_velocity_RminL[velInd],linestyle='None')

        jitter = jitterSft*np.random.rand(np.shape(spIndsAll)[0],np.shape(spIndsAll)[1])-jitterSft/2 # add some random jitter to x-placement of single trial responses
        ax[2,0].plot(spInds,np.nanmean(postwind_nonflyingNorm_allFlies[:,:,velInd],axis=0),marker='.',color=const.colors_velocity_RminL[velInd])
        ax[2,0].plot(spIndsAll+jitter,postwind_nonflyingNorm_allFlies[:,:,velInd],marker='.',color=const.colors_velocity_RminL[velInd],linestyle='None')

        jitter = jitterSft*np.random.rand(np.shape(spIndsAll)[0],np.shape(spIndsAll)[1])-jitterSft/2 # add some random jitter to x-placement of single trial responses
        ax[2,1].plot(spInds,np.nanmean(postwind_flyingNorm_allFlies[:,:,velInd],axis=0),marker='.',color=const.colors_velocity_RminL[velInd])
        ax[2,1].plot(spIndsAll+jitter,postwind_flyingNorm_allFlies[:,:,velInd],marker='.',color=const.colors_velocity_RminL[velInd],linestyle='None')

        ax[0,0].text(3,6-velInd*shiftAmt-shiftAmt*3, str(speeds[velInd])+' cm/s', color=const.colors_velocity_RminL[velInd])

        jitter = jitterSft*np.random.rand(np.shape(spIndsAll)[0],np.shape(spIndsAll)[1])-jitterSft/2 # add some random jitter to x-placement of single trial responses
        ax2.plot(spInds,np.nanmean(wind_allNorm_allFlies[:,:,velInd],axis=0),marker='.',color=const.colors_velocity_RminL[velInd])
        ax2.plot(spIndsAll+jitter,wind_allNorm_allFlies[:,:,velInd],marker='.',color=const.colors_velocity_RminL[velInd],linestyle='None')


    ax[0,0].set_ylabel('pre-wind',color=const.axisColor)
    ax[1,0].set_ylabel('during wind',color=const.axisColor)
    ax[2,0].set_ylabel('post-wind',color=const.axisColor)
    ax2.set_ylabel('number of movements\nnormalized by # trials',color=const.axisColor)

    for ii in range(3): # for each pre,during,post-wind condition plot normalized counts (rows)
        for jj in range(2): # plot nonflying and flying counts (columns)
            ax[ii,jj].set_facecolor(const.figColor)
            ax[ii,jj].spines['right'].set_visible(False)
            ax[ii,jj].spines['top'].set_visible(False)
            ax[ii,jj].spines['left'].set_color(const.axisColor)
            ax[ii,jj].spines['bottom'].set_color(const.axisColor)
            ax[ii,jj].tick_params(direction='in', length=5, width=0.5)
            ax[ii,jj].tick_params(axis='y',colors=const.axisColor)
            # configure the y-axis
            ax[ii,jj].set_ylim([-0.75,10])
            ax[ii,jj].spines['left'].set_bounds(0, 10) #do not extend y-axis line beyond ticks
            # configure the x-axis
            ax[ii,jj].set_xticks(np.arange(5))
            ax[ii,jj].set_xlim([-0.5,4.5])
            ax[ii,jj].tick_params(axis='x',colors=const.axisColor,rotation=30)
            ax[ii,jj].spines['bottom'].set_bounds(0, 4) #do not extend x-axis line beyond ticks
            # ax[ii].set_xlabel('Wind direction (deg)',color=const.axisColor)
            if ii == 2:
                ax[ii,jj].set_xticklabels(const.windDirLabelsIC)
            else:
                ax[ii,jj].set_xticklabels('')

    ax2.set_facecolor(const.figColor)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_color(const.axisColor)
    ax2.spines['bottom'].set_color(const.axisColor)
    ax2.tick_params(direction='in', length=5, width=0.5)
    ax2.tick_params(axis='y',colors=const.axisColor)
    # configure the y-axis
    ax2.set_ylim([-0.75,6])
    ax2.spines['left'].set_bounds(0, 6) #do not extend y-axis line beyond ticks
    # configure the x-axis
    ax2.set_xticks(np.arange(5))
    ax2.set_xlim([-0.5,4.5])
    ax2.tick_params(axis='x',colors=const.axisColor,rotation=30)
    ax2.spines['bottom'].set_bounds(0, 4) #do not extend x-axis line beyond ticks
    ax2.set_xticklabels(const.windDirLabelsIC)

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig: # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        # save raster (cross-fly)
        savepath = figPath + expt + '_numActiveMovementVsDirSecondSeg' + '.png'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + expt + '_numActiveMovementVsDirSecondSeg' + '.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + expt + '_numActiveMovementVsDirSecondSegAllTrials' + '.png'
        fig2.savefig(savepath, facecolor=fig2.get_facecolor())
        savepath = figPath + expt + '_numActiveMovementVsDirSecondSegAllTrials' + '.pdf'
        fig2.savefig(savepath, facecolor=fig2.get_facecolor())

# Plot paired nolight-light data for Gtacr1 inactivation experiments
# Plotting for non-flying trials (no partial and no fully flying trials).
# As a result, for some flies, we will not have paired data (some of the flies few during several trials!)
def plot_inactivation_paired_active_movements(expt='18D07_inactivate', cameraView='frontal',importAnew=0,savefig=0,secondSeg=1):
    # grab active movement data
    if 'inactivate' in expt:
        print('plot inactivation data!')
        [_, _, wind_nonflyingNorm_allFlies_noLight, _,_, _,_,_] = get_movement_count_during_wind(expt,cameraView,2,importAnew,savefig,secondSeg)
        [_, _, wind_nonflyingNorm_allFlies_withLight, _,_, _,_,_] = get_movement_count_during_wind(expt,cameraView,1,importAnew,savefig,secondSeg)
        velsToPlot = [0,1] #speeds = [0, 200] #only ran two speeds, 0 and 200 cm/s on these experiments
    else:
        [_, _, wind_nonflyingNorm_allFlies_noLight, _,_, _,_,_] = get_movement_count_during_wind(expt,cameraView,0,importAnew,savefig,secondSeg)
        [_, _, wind_nonflyingNorm_allFlies_withLight, _,_, _,_,_] = get_movement_count_during_wind(expt,cameraView,0,importAnew,savefig,secondSeg)
        velsToPlot = [0,3] #speeds = [0, 200] #0 and 200 cm/s on these experiments

    print(np.shape(wind_nonflyingNorm_allFlies_noLight))

    # plot legend for speeds (HARD-CODED)
    shiftAmt = 0.5
    jitterSft = 0.2
    velIndsColor = [3,1]

    sftAmt = 0.2
    yMax = 4.9
    sft = 0.3
    compInds = [0,1]

    # plot individual trials with integer values only is hard to look at)
    fig, ax = plt.subplots(1,3,facecolor=const.figColor,figsize=(6,6))
    fig.suptitle(expt+'\nActive movements\nduring wind (all directions)',color=const.axisColor)

    for ii,velInd in enumerate(velsToPlot):
        allNoLight = np.nanmean(wind_nonflyingNorm_allFlies_noLight[:,:,velInd],axis=1) # combine data across all directions
        meanNoLight = np.nanmean(allNoLight,axis=0)
        allLight = np.nanmean(wind_nonflyingNorm_allFlies_withLight[:,:,velInd],axis=1) # combine data across all directions
        meanLight = np.nanmean(allLight,axis=0)
        # compute different and standard deviation for reporting result (and plot pVal later as well)
        meanLightDiff = np.nanmean(allLight-allNoLight)
        stdLightDiff = np.std(allLight-allNoLight)

        ax[ii].plot([allNoLight, allLight],marker='None',color=const.axisColor,linestyle='-',linewidth=1)
        ax[ii].plot([0,1],[meanNoLight, meanLight],marker='None',color=const.axisColor,linestyle='-',linewidth=3)
        # compute and plot result of paired ttest between no light and light conditions for same flies
        #0 cm/s
        noLight = np.nanmean(wind_nonflyingNorm_allFlies_noLight[:,:,velInd],axis=1) # combine data across all directions
        light = np.nanmean(wind_nonflyingNorm_allFlies_withLight[:,:,velInd],axis=1) # combine data across all directions
        #if expt == '18D07_inactivate':
        result1 = stats.ttest_rel(noLight, light, nan_policy='omit') #ttest for non-nan flies
        pVal = [result1.pvalue]
        print(pVal[0])
        pVal = pVal[0]

        #for ii in range(np.shape(pVals)[0]):
        yy = yMax-sftAmt
        ax[ii].plot([0,1],[yy,yy], color=const.axisColor, linewidth=1)
        if pVal < 0.001:
            mkr = '***'
        elif pVal < 0.01:
            mkr = '**'
        elif pVal < 0.05:
            mkr = '*'
        else: mkr = 'ns'
        ax[ii].text(0.5, yy+0.1, mkr,
            color=const.axisColor, fontsize=const.fontSize_axis+1)
                       # configure axes
        ax[ii].set_facecolor(const.figColor)
        ax[ii].spines['right'].set_visible(False)
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['left'].set_color(const.axisColor)
        ax[ii].spines['bottom'].set_visible(False)
        ax[ii].tick_params(direction='in', length=5, width=0.5)
        ax[ii].tick_params(axis='y', colors=const.axisColor)
        ax[ii].set_ylabel('average # active movements\nduring wind', color=const.axisColor,
                         fontsize=const.fontSize_axis)
        # configure the y-axis
        ax[ii].set_ylim([-0.5, 5.5])
        ax[ii].set_yticks([0,1,2,3,4, 5])
        ax[ii].spines['left'].set_bounds(0, 5)  # do not extend y-axis line beyond ticks
        ax[ii].set_xlim([-0.25, 1.25])
        ax[ii].xaxis.set_ticks([])
        if 'inactivate' in expt:
            ax[ii].text(0, -0.5, 'no light', rotation=0, horizontalalignment='center', color=const.axisColor)
            ax[ii].text(1, -0.5, 'with light', rotation=0, horizontalalignment='center', color=const.axisColor)
        else:
            ax[ii].text(0, -0.5, '2nd seg', rotation=0, horizontalalignment='center', color=const.axisColor)
            ax[ii].text(1, -0.5, '3rd seg', rotation=0, horizontalalignment='center', color=const.axisColor)

        # plot p-values, differences in averages)

        ax[2].set_facecolor(const.figColor)
        if velInd == 0:
            ax[2].axis('off')
            ax[2].text(0.005, 0.9, 'pVal 0 cm/s: '+str(round(pVal,3)), rotation=0, horizontalalignment='left', color=const.axisColor)
            diff = meanLight-meanNoLight
            ax[2].text(0.005, 0.8, 'avg(light-no light)='+str(round(meanLightDiff,3)), rotation=0, horizontalalignment='left', color=const.axisColor)
            ax[2].text(0.005, 0.7, 'std(light-no light)='+str(round(stdLightDiff,3)), rotation=0, horizontalalignment='left', color=const.axisColor)
        else:
            ax[2].text(0.005, 0.5, 'pVal 200 cm/s: '+str(round(pVal,3)), rotation=0, horizontalalignment='left', color=const.axisColor)
            ax[2].text(0.005, 0.4, 'avg(light-no light)='+str(round(meanLightDiff,3)), rotation=0, horizontalalignment='left', color=const.axisColor)
            ax[2].text(0.005, 0.3, 'std(light-no light)='+str(round(stdLightDiff,3)), rotation=0, horizontalalignment='left', color=const.axisColor)

    ax[1].text(0.5, 5.1, '200 cm/s', rotation=0, horizontalalignment='center', color=const.axisColor)
    ax[0].text(0.5, 5.1, '0 cm/s', rotation=0, horizontalalignment='center', color=const.axisColor)

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig: # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        if 'inactivate' not in expt:
            txt = '_secondVsThird'
        else:
            txt = ''
        # save raster (cross-fly)
        savepath = figPath + expt + '_pairedInactivationMovements' + txt+'.png'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + expt + '_pairedInactivationMovements' + txt+'.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())


# Plot raw antenna traces around active movements
def plot_active_movement_traces_odor(expt='2021_09_09_E1', cameraView='frontal', plotTracesSeparately = 0,plotInactivate=0, savefig=0):
    TEST_TWO_DIRS = 0
    INCLUDE_ON_OFFSET = 0
    exptNotes = get_mat_notes(expt)
    yaxis_min = -25
    yaxis_max = 25
    baselineLength = 5 # 5 samples (or the probably 30 used in this window)
    sft = 5

    windOnInd = const.windStart*const.framerate-3
    windOffInd = (const.windStart+const.windTime)*const.framerate
    windowExclusion = 12
    windOnsetInds = list(range(windOnInd,windOnInd+windowExclusion))
    windOffsetInds = list(range(windOffInd,windOffInd+windowExclusion))
    peaksAll, allTraces, allDirs, allVels = get_active_movements(expt, cameraView='frontal',allowFlight = 1,TEST=TEST_TWO_DIRS,importAnew=0,savefig=0)
    antInds = [1,3] # look at third segments of right and left antenna
    dirInds = np.unique(allDirs)

    # make two plots: one with traces overlaid, one with them plotted in a staggered fashion
    fig, axTrace = plt.subplots(2,len(dirInds),facecolor=const.figColor,figsize=(10,7))
    fig2, axTraceSep = plt.subplots(2,len(dirInds),facecolor=const.figColor,figsize=(10,16))

    # MxNxO matrix where m=number of active movements, N=number of antenna segments, O = framerate
    maxPossiblePeaks = int(np.sum(peaksAll[:,0,:])+np.sum(peaksAll[:,1,:])) #sum indicates #of raw movements
    allActiveTraces = [np.nan] * np.zeros((maxPossiblePeaks, np.shape(antInds)[0],const.framerate))
    allDirections = [np.nan] * np.zeros((maxPossiblePeaks, np.shape(antInds)[0]))
    pkNum = -1

    for trialNum in range(np.shape(peaksAll)[0]):
        # plot each antenna part (e.g. left and right arista)
        for antIdx, antNum in enumerate(antInds):
            peaks = np.squeeze(np.where(peaksAll[trialNum,antIdx,:]>0))

            # find peaks that are in onset/offset regions
            if (np.shape(peaks)):
                onsetPeaks = [x for x in peaks if x in windOnsetInds]
                if onsetPeaks != []:
                    onsetRidPeak = np.squeeze(np.where(peaks == onsetPeaks))
                else: onsetRidPeak = []

                offsetPeaks = [x for x in peaks if x in windOffsetInds]
                if offsetPeaks != []:
                    offsetRidPeak = np.squeeze(np.where(peaks == offsetPeaks))
                else: offsetRidPeak = []

            direction = np.squeeze(np.where(int(allDirs[trialNum])==dirInds))

            if (np.shape(peaks)):
                for ii, peak in enumerate(peaks):
                    start = peak-const.peakWindow
                    stop  = peak+const.peakWindow
                    startBase = start
                    stopBase = start+baselineLength
                    baseline = np.nanmean(allTraces[trialNum,startBase:stopBase,antIdx])
                    tt = allTraces[trialNum,start:stop,antIdx]-baseline #bseline subtract the trace to align
                    # plot movements not at wind on/offset (most of them!)
                    if (stop<const.numFrames) & (start>0) & ~(ii in onsetRidPeak) & ~(ii in offsetRidPeak): #take traces not overlapping with edge of trial (very beginning or very end)
                        pkNum = pkNum+1
                        allActiveTraces[pkNum, antIdx,:] = tt
                        allDirections[pkNum, antIdx] = int(direction)
                        axTrace[antIdx,direction].plot(tt,color=const.colors_antAngs[antNum], alpha=const.transparencyActiveTrace)
                        axTraceSep[antIdx,direction].plot(tt+sft*trialNum+sft*ii,color=const.colors_antAngs[antNum], alpha=const.transparencyAntTrace)

                    #plot movements right after start of wind
                    elif (stop<const.numFrames) & (start>0) & (ii in onsetRidPeak) & INCLUDE_ON_OFFSET:
                        axTrace[antIdx,direction].plot(tt,color=const.windOnColor, alpha=const.transparencyActiveTrace)
                        axTraceSep[antIdx,direction].plot(tt+sft*trialNum+sft*ii,const.windOnColor, alpha=const.transparencyAntTrace)
                    #plot movements right after stop of wind
                    elif (stop<const.numFrames) & (start>0) & (ii in offsetRidPeak) & INCLUDE_ON_OFFSET:
                        axTrace[antIdx,direction].plot(tt,color=const.windOffColor, alpha=const.transparencyActiveTrace)
                        axTraceSep[antIdx,direction].plot(tt+sft*trialNum+sft*ii,color=const.windOffColor, alpha=const.transparencyAntTrace)

    # set background color, labels for figures:
    for ii, antNum in enumerate(antInds):
        for jj, dir in enumerate(dirInds):
            axTrace[ii,jj].set_facecolor(const.figColor)
            axTrace[ii,jj].set_ylim(yaxis_min, yaxis_max)
            axTrace[ii,jj].axis('off')

            axTraceSep[ii,jj].set_facecolor(const.figColor)
            axTraceSep[ii,jj].set_ylim([-sft,(np.shape(peaksAll)[0]+10)*sft])
            axTraceSep[ii,jj].axis('off')
            if (jj == 0) & (ii == 0):
                axTrace[ii,jj].set_title('no wind',color=const.axisColor)
                axTraceSep[ii,jj].set_title('no wind',color=const.axisColor)
            elif ii == 0:
                axTrace[ii,jj].set_title(str(const.windOdorNames[int(dir)-1]),color=const.axisColor)
                axTraceSep[ii,jj].set_title(str(const.windOdorNames[int(dir)-1]),color=const.axisColor)

    # plot legend for how each kind of motion is color-coded
    for antIdx, antNum in enumerate(antInds):
        axTrace[0,0].text(0,sft*antIdx,const.angPairNames[cameraView][antNum],color=const.colors_antAngs[antNum])
        axTraceSep[0,0].text(0,2*sft*antIdx,const.angPairNames[cameraView][antNum],color=const.colors_antAngs[antNum])
    axTrace[0,0].text(0,-sft*1,'after wind on',color=const.windOnColor)
    axTrace[0,0].text(0,-sft*2,'after wind off',color=const.windOffColor)
    axTraceSep[0,0].text(0,-2*sft*1,'after wind on',color=const.windOnColor)
    axTraceSep[0,0].text(0,-2*sft*2,'after wind off',color=const.windOffColor)

    fig.suptitle('raw antenna angles (overlaid) '+expt,color=const.axisColor)
    fig2.suptitle('raw antenna angles (separated) '+expt,color=const.axisColor)

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        savepath = figPath + 'rawActiveTracesOverlaid_' + expt + '.png'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + 'rawActiveTracesSeparated_' + expt + '.png'
        fig2.savefig(savepath, facecolor=fig2.get_facecolor())

    return allActiveTraces, allDirections

# Plot raw antenna traces around active movements for inactivation data
def plot_active_movement_traces_inactivation(expt='2021_11_12_E1', cameraView='frontal', plotTracesSeparately = 0,plotInactivate=1, savefig=0):
    INCLUDE_ON_OFFSET = 0
    activate = get_activation_trials(expt)
    yaxis_min = -25
    yaxis_max = 25
    baselineLength = 5  # 5 samples (or the probably 30 used in this window)
    sft = 5

    windOnInd = const.windStart*const.framerate-3
    windOffInd = (const.windStart+const.windTime)*const.framerate
    windowExclusion = 12
    windOnsetInds = list(range(windOnInd, windOnInd+windowExclusion))
    windOffsetInds = list(range(windOffInd, windOffInd+windowExclusion))
    peaksAll, allTraces, allDirs, allVels = get_active_movements(expt, cameraView='frontal', allowFlight = 1, TEST=0, importAnew=0, savefig=0);

    antInds = [1, 3]  # look at third segments of right and left antenna
    dirInds = np.unique(allDirs)
    velocities = np.unique(allVels)

    # make two plots: one with traces overlaid, one with them plotted in a staggered fashion
    fig, axTrace = plt.subplots(2, len(dirInds), facecolor=const.figColor, figsize=(10,7))
    fig2, axTraceSep = plt.subplots(2, len(dirInds), facecolor=const.figColor, figsize=(10,16))

    # MxNxO matrix where m=number of active movements, N=number of antenna segments, O = framerate
    maxPossiblePeaks = int(np.sum(peaksAll[:,0,:])+np.sum(peaksAll[:,1,:])) #sum indicates #of raw movements
    allActiveTraces = [np.nan] * np.zeros((maxPossiblePeaks, np.shape(antInds)[0],const.framerate))
    allDirections = [np.nan] * np.zeros((maxPossiblePeaks, np.shape(antInds)[0]))
    allVelocities = [np.nan] * np.zeros((maxPossiblePeaks, np.shape(antInds)[0]))
    pkNum = -1

    for trialNum in range(np.shape(peaksAll)[0]):
        if ((plotInactivate==2) & (activate[trialNum]==0)) | (plotInactivate==0) | ((plotInactivate==1) & (activate[trialNum]==1)):
            # plot each antenna part (e.g. left and right arista)
            for antIdx, antNum in enumerate(antInds):
                peaks = np.squeeze(np.where(peaksAll[trialNum,antIdx,:]>0))

                # find peaks that are in onset/offset regions
                if (np.shape(peaks)):
                    onsetPeaks = [x for x in peaks if x in windOnsetInds]
                    if onsetPeaks != []:
                        onsetRidPeak = np.squeeze(np.where(peaks == onsetPeaks))
                    else: onsetRidPeak = []

                    offsetPeaks = [x for x in peaks if x in windOffsetInds]
                    if offsetPeaks != []:
                        offsetRidPeak = np.squeeze(np.where(peaks == offsetPeaks))
                    else: offsetRidPeak = []

                direction = np.squeeze(np.where(int(allDirs[trialNum])==dirInds))
                velocity = np.squeeze(np.where(int(allVels[trialNum])==velocities))

                if (np.shape(peaks)):
                    for ii, peak in enumerate(peaks):
                        start = peak-const.peakWindow
                        stop  = peak+const.peakWindow
                        startBase = start
                        stopBase = start+baselineLength
                        baseline = np.nanmean(allTraces[trialNum,startBase:stopBase,antIdx])
                        tt = allTraces[trialNum,start:stop,antIdx]-baseline #bseline subtract the trace to align
                        # plot movements not at wind on/offset (most of them!)
                        if (stop<const.numFrames) & (start>0) & ~(ii in onsetRidPeak) & ~(ii in offsetRidPeak): #take traces not overlapping with edge of trial (very beginning or very end)
                            pkNum = pkNum+1
                            allActiveTraces[pkNum, antIdx,:] = tt
                            allDirections[pkNum, antIdx] = int(direction)
                            allVelocities[pkNum, antIdx] = int(velocity)
                            axTrace[antIdx,direction].plot(tt,color=const.colors_antAngs[antNum], alpha=const.transparencyActiveTrace)
                            axTraceSep[antIdx,direction].plot(tt+sft*trialNum+sft*ii,color=const.colors_antAngs[antNum], alpha=const.transparencyAntTrace)

                        #plot movements right after start of wind
                        elif (stop<const.numFrames) & (start>0) & (ii in onsetRidPeak) & INCLUDE_ON_OFFSET:
                            axTrace[antIdx,direction].plot(tt,color=const.windOnColor, alpha=const.transparencyActiveTrace)
                            axTraceSep[antIdx,direction].plot(tt+sft*trialNum+sft*ii,const.windOnColor, alpha=const.transparencyAntTrace)
                        #plot movements right after stop of wind
                        elif (stop<const.numFrames) & (start>0) & (ii in offsetRidPeak) & INCLUDE_ON_OFFSET:
                            axTrace[antIdx,direction].plot(tt,color=const.windOffColor, alpha=const.transparencyActiveTrace)
                            axTraceSep[antIdx,direction].plot(tt+sft*trialNum+sft*ii,color=const.windOffColor, alpha=const.transparencyAntTrace)

    # set background color, labels for figures:
    for ii, antNum in enumerate(antInds):
        for jj, dir in enumerate(dirInds):
            axTrace[ii,jj].set_facecolor(const.figColor)
            axTrace[ii,jj].set_ylim(yaxis_min, yaxis_max)
            axTrace[ii,jj].axis('off')
            axTraceSep[ii,jj].set_facecolor(const.figColor)
            axTraceSep[ii,jj].set_ylim([-sft,(np.shape(peaksAll)[0]+10)*sft])
            axTraceSep[ii,jj].axis('off')
            axTrace[ii,jj].set_title(str(const.windDirNames[int(jj)]),color=const.axisColor)
            axTraceSep[ii,jj].set_title(str(const.windDirNames[int(jj)]),color=const.axisColor)

    # plot legend for how each kind of motion is color-coded
    for antIdx, antNum in enumerate(antInds):
        axTrace[0,0].text(0,sft*antIdx,const.angPairNames[cameraView][antNum],color=const.colors_antAngs[antNum])
        axTraceSep[0,0].text(0,2*sft*antIdx,const.angPairNames[cameraView][antNum],color=const.colors_antAngs[antNum])
    axTrace[0,0].text(0,-sft*1,'after wind on',color=const.windOnColor)
    axTrace[0,0].text(0,-sft*2,'after wind off',color=const.windOffColor)
    axTraceSep[0,0].text(0,-2*sft*1,'after wind on',color=const.windOnColor)
    axTraceSep[0,0].text(0,-2*sft*2,'after wind off',color=const.windOffColor)

    if plotInactivate == 1: extraText = 'lightOn'
    elif plotInactivate == 2: extraText = 'noLight'
    else: extraText = ''
    fig.suptitle('raw antenna angles (overlaid) '+expt + ' '+extraText,color=const.axisColor)
    fig2.suptitle('raw antenna angles (separated) '+expt + ' '+extraText,color=const.axisColor)

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        savepath = figPath + 'rawActiveTracesOverlaid_' + expt + '.png'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + 'rawActiveTracesSeparated_' + expt + '.png'
        fig2.savefig(savepath, facecolor=fig2.get_facecolor())

    return allActiveTraces, allDirections, allVelocities

# Hard-coded: set preferences for autocorrelation plots
def set_axis_prefs_autocorr(ax, maxlag):
    ax.set_facecolor(const.figColor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(const.axisColor)
    ax.spines['bottom'].set_color(const.axisColor)
    ax.tick_params(direction='in', length=5, width=0.5)
    ax.tick_params(axis='y',colors=const.axisColor)

    #configure the y-axis
    ax.spines['left'].set_bounds(-0.2,1) #do not extend y-axis line beyond ticks
    ax.set_ylim(-0.3, 1)

    #configure the x-axis
    ax.set_xticks([0, maxlag, maxlag*2])
    ax.spines['bottom'].set_bounds(0,maxlag*2) #do not extend x-axis line beyond ticks
    ax.set_xticklabels(['-1','0','1'])
    ax.set_xlabel('lag (s)')
    ax.tick_params(axis='x',colors=const.axisColor)

# grab traces from all the odor experiments and plot
# warning: some serious HARD CODING in here!
def plot_active_traces_overlaid_odor(expt='CS_odor', cameraView='frontal', savefig=0):
    PLOT_AVG = 0
    PLOT_STD = 0
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)

    antInds = [1,3] # look at third segments of right and left antenna
    dirInds = [0,3,4,7,8]

    fig, axTrace = plt.subplots(2,len(dirInds),facecolor=const.figColor,figsize=(10,7))
    yaxis_min = -25
    yaxis_max = 25

    count = 0
    maxlag = 59
    autoCorrAllFlies = np.ones((np.shape(allExpts)[0],np.shape(antInds)[0],np.shape(dirInds)[0],maxlag*2+1))
    autoCorrAllFlies[:] = np.nan

    for noteInd, notes in enumerate(allExpts):
        flyExpt = notes['date']
        allActiveTraces, allDirections = plot_active_movement_traces_odor(expt=flyExpt, cameraView='frontal', plotTracesSeparately = 0,savefig=savefig)
        if 'allTraces' not in locals():
            allTraces = allActiveTraces
            allDirs = allDirections
        else:
            allTraces = np.vstack((allTraces,allActiveTraces))
            allDirs = np.vstack((allDirs,allDirections))

        # compute autocorrelation for each fly here:
        autoCorrThisFly = np.ones((np.shape(allActiveTraces)[0],np.shape(antInds)[0],np.shape(dirInds)[0],maxlag*2+1))
        autoCorrThisFly[:] = np.nan
        # go through every trace for this fly and compute autoCorr
        for traceNum in range(np.shape(allActiveTraces)[0]):
            for antIdx, antNum in enumerate(antInds):
                dirIdx = allDirections[traceNum,antIdx]#activeDirections[traceNum]
                if ~np.isnan(dirIdx):
                    #trace = activeTraces[traceNum,:]
                    trace = allActiveTraces[traceNum,antIdx,:]
                    xx = plt.acorr(trace-np.nanmean(trace),normed=True,usevlines=False,maxlags=maxlag) #xx[1] contains correlation vector
                    autoCorrThisFly[traceNum,antIdx,int(dirIdx):] = xx[1]
        # compute autocorr average for each fly and tally up (plot later!)
        autoCorrAllFlies[noteInd,:,:,:] = np.nanmean(autoCorrThisFly,0)

    print(str(np.shape(allTraces)[0])+' raw average active movement traces')
    print('from ' + str(np.shape(allExpts)[0]) + ' flies')

    # plot all the individual fly raw traces
    for traceNum in range(np.shape(allActiveTraces)[0]):
        for antIdx, antNum in enumerate(antInds):
            dirIdx = allDirections[traceNum,antIdx]
            if ~np.isnan(dirIdx):
                trace = allActiveTraces[traceNum,antIdx,:]
                axTrace[antIdx,int(dirIdx)].plot(trace,
                    color=const.colors_antAngs[antNum], alpha=const.transparencyAllActiveTraces)

    # Plot auto correlogram for all flies (and plot average)
    figAC, axTraceAC = plt.subplots(np.shape(antInds)[0],len(dirInds),figsize=(10,7),facecolor=const.figColor)
    figAC.suptitle('Autocorrelation average for single (green) and across flies (gray)',color=const.axisColor)#, nlags='+str(maxlag))
    for ai in range(np.shape(antInds)[0]):
        # plot each fly's average autocorr
        for ii, dir in enumerate(dirInds):
            for flyNum in range(np.shape(autoCorrAllFlies)[0]):
                axTraceAC[ai,ii].plot(autoCorrAllFlies[flyNum,ai,ii,:],color='green')
                halfMaxWid = np.squeeze(np.where(autoCorrAllFlies[flyNum,ai,ii,:]>0.5))
                if np.shape(halfMaxWid):
                    if np.size(halfMaxWid)>1:
                        halfMaxWid = halfMaxWid[-1]/const.framerate
                    else:
                        halfMaxWid = halfMaxWid/const.framerate
                    print('Half max width (s) ' + str(halfMaxWid))
            #plot average autocorr across flies
            crossFlyMeanAutocorr = np.nanmean(autoCorrAllFlies[:,ai,ii,:],0)
            axTraceAC[ai,ii].plot(crossFlyMeanAutocorr,color='gray')
            set_axis_prefs_autocorr(axTraceAC[ai,ii], maxlag) #set axis prefs

            if (ii == 0):
                axTraceAC[ai,ii].set_title('no wind',color=const.axisColor)
            else:
                axTraceAC[ai,ii].set_title(str(const.windOdorNames[int(dir)-1]),color=const.axisColor)
            if ii == 0:
                axTraceAC[ai,ii].set_ylabel('autocorrelation'+const.angPairNames[cameraView][antInds[ai]],color=const.axisColor)

    # plot autocorr averages on top of one another for comparison
    # right now, hard-coded: odor/no odor +45
    # Plot auto correlogram for all flies (and plot average)
    figIpsiContraCorr, corrAvgComp = plt.subplots(1,2,figsize=(10,3),facecolor=const.figColor)
    corrAvgComp[0].set_title('ipsi wind (gray) vs. odor (blue)',color=const.axisColor)
    corrAvgComp[1].set_title('contra',color=const.axisColor)
    # ipsi 45
    crossFlyMeanAutocorr_ipsiR = np.nanmean(autoCorrAllFlies[:,0,3,:],0) #+45
    crossFlyMeanAutocorr_ipsiL = np.nanmean(autoCorrAllFlies[:,1,1,:],0) #+45
    crossFlyMeanAutocorr_ipsi = np.nanmean([crossFlyMeanAutocorr_ipsiR, crossFlyMeanAutocorr_ipsiL],0)
    corrAvgComp[0].plot(crossFlyMeanAutocorr_ipsi,color='gray')
    # ipsi odor 45
    crossFlyMeanAutocorr_ipsiOdorR = np.nanmean(autoCorrAllFlies[:,0,4,:],0) #odor +45
    crossFlyMeanAutocorr_ipsiOdorL = np.nanmean(autoCorrAllFlies[:,1,2,:],0) #odor +45
    crossFlyMeanAutocorr_ipsiOdor = np.nanmean([crossFlyMeanAutocorr_ipsiOdorR, crossFlyMeanAutocorr_ipsiOdorL],0) #odor +45
    corrAvgComp[0].plot(crossFlyMeanAutocorr_ipsiOdor,color='blue')

    # contra 45
    crossFlyMeanAutocorr_contraR = np.nanmean(autoCorrAllFlies[:,0,1,:],0) #+45
    crossFlyMeanAutocorr_contraL = np.nanmean(autoCorrAllFlies[:,1,3,:],0) #+45
    crossFlyMeanAutocorr_contra = np.nanmean([crossFlyMeanAutocorr_contraR, crossFlyMeanAutocorr_contraL],0)
    corrAvgComp[1].plot(crossFlyMeanAutocorr_contra,color='gray')
    # contra odor 45
    crossFlyMeanAutocorr_contraOdorR = np.nanmean(autoCorrAllFlies[:,0,2,:],0) #odor +45
    crossFlyMeanAutocorr_contraOdorL = np.nanmean(autoCorrAllFlies[:,1,4,:],0) #odor +45
    crossFlyMeanAutocorr_contraOdor = np.nanmean([crossFlyMeanAutocorr_contraOdorR, crossFlyMeanAutocorr_contraOdorL],0) #odor +45
    corrAvgComp[1].plot(crossFlyMeanAutocorr_contraOdor,color='blue')

    set_axis_prefs_autocorr(corrAvgComp[0], maxlag) #set axis prefs
    set_axis_prefs_autocorr(corrAvgComp[1], maxlag) #set axis prefs

    # plot average trace
    allDirs = np.squeeze(allDirs)
    if PLOT_AVG:
        for antIdx, antNum in enumerate(antInds):
            for dirIdx, dir in enumerate(dirInds):
                thisDirInds = np.squeeze(np.where(allDirs==dirIdx))
                average = np.nanmean(allTraces[thisDirInds,:],0)
                stddev = np.nanstd(allTraces[thisDirInds,:],0)
                axTrace[antIdx,int(dirIdx)].plot(average,
                    color=const.colors_antAngs[antNum],linewidth=3)
                if PLOT_STD:
                    axTrace[antIdx,int(dirIdx)].plot(stddev,
                        color='green',linewidth=3)

    # set background color, labels for figures:
    scaleBarSize = 10
    scaleWidth = 2
    scaleX = 35
    scaleY = 1
    for ii, antNum in enumerate(antInds):
        for jj, dir in enumerate(dirInds):
            axTrace[ii,jj].set_facecolor(const.figColor)
            axTrace[ii,jj].set_ylim(yaxis_min, yaxis_max)
            axTrace[ii,jj].axis('off')

            if (jj == 0) & (ii == 0):
                axTrace[ii,jj].set_title('no wind',color=const.axisColor)
                # add scale bar (angle, in Y)
                axTrace[ii,jj].add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
                    facecolor = const.axisColor))
                axTrace[ii,jj].text(scaleX+scaleWidth*2, scaleY+scaleBarSize/2,
                    str(scaleBarSize) + const.degree_sign,color=const.axisColor,
                    fontsize=const.fontSize_angPair,horizontalalignment='left',
                    verticalalignment='center')
            elif ii == 0:
                axTrace[ii,jj].set_title(str(const.windOdorNames[int(dir)-1]),color=const.axisColor)

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        if PLOT_STD:
            extraText = 'withStd'
        elif PLOT_AVG:
            extraText = 'withAvg'
        else:
            extraText = 'noAvg'
        savepath = figPath + 'rawActiveTracesAllFlies_' + expt +'_'+ extraText+ '.png'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + 'rawActiveTracesAllFlies_' + expt + '_'+extraText+ '.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        # save autocorrelation plot with single-fly averages
        savepath = figPath + 'activeTraces_autocorr_' + expt + '.png'
        figAC.savefig(savepath, facecolor=figAC.get_facecolor())
        savepath = figPath + 'activeTraces_autocorr_' + expt + '.pdf'
        figAC.savefig(savepath, facecolor=figAC.get_facecolor())
        # save average autocorrelation comparison plot (ipsi vs. contra, wind vs. odor)

        savepath = figPath + 'autocorr_avg_wind_odor' + expt + '.png'
        figIpsiContraCorr.savefig(savepath, facecolor=figIpsiContraCorr.get_facecolor())
        savepath = figPath + 'autocorr_avg_wind_odor' + expt + '.pdf'
        figIpsiContraCorr.savefig(savepath, facecolor=figIpsiContraCorr.get_facecolor())
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()


# grab traces from all the inactivation experiments and plot (18D07 or 91F02)
# warning: some serious HARD CODING in here!
def plot_active_traces_overlaid_inactivation(expt='18D07_inactivate', cameraView='frontal', importAnew = 0, savefig=0):
    PLOT_AVG = 0
    PLOT_STD = 0
    import loadNotes_mnOptoVelocityTuning as ln
    notesName = 'notes_'+expt
    allExpts = eval('ln.'+notesName)

    antInds = [1,3] # look at third segments of right and left antenna
    dirInds = [0,1,2,3,4]
    velInds = [0,1] #this is hard-coded for 0 and 200 cm/s (need to modify if we want to use outside of inactivation data)

    count = 0
    maxlag = 15#
    autoCorrAllFlies_lightOn = np.ones((np.shape(allExpts)[0],np.shape(antInds)[0],np.shape(dirInds)[0],maxlag*2+1))
    autoCorrAllFlies_lightOn[:] = np.nan
    autoCorrAllFlies_noLight = np.ones((np.shape(allExpts)[0],np.shape(antInds)[0],np.shape(dirInds)[0],maxlag*2+1))
    autoCorrAllFlies_noLight[:] = np.nan

    for noteInd, notes in enumerate(allExpts):
        flyExpt = notes['date']
        # Save data structures for faster plowwing
        saved_allActiveTraces_lightOn_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allActiveTraces_lightOn.npy'
        saved_allDirections_lightOn_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allDirections_lightOn.npy'
        saved_allVelocities_lightOn_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allVelocities_lightOn.npy'
        saved_allActiveTraces_noLight_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allActiveTraces_noLight.npy'
        saved_allDirections_noLight_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allDirections_noLight.npy'
        saved_allVelocities_noLight_fn = const.savedDataDirectory+flyExpt+'_'+cameraView+'_'+'allVelocities_noLight.npy'

        # check if the notes have been saved previously, then load
        if (importAnew==0) & (os.path.isfile(saved_allActiveTraces_lightOn_fn)==True):
            # load previously saved active movement data
            allActiveTraces_lightOn = np.load(saved_allActiveTraces_lightOn_fn)
            allDirections_lightOn = np.load(saved_allDirections_lightOn_fn)
            allVelocities_lightOn = np.load(saved_allVelocities_lightOn_fn)
            allActiveTraces_noLight = np.load(saved_allActiveTraces_noLight_fn)
            allDirections_noLight = np.load(saved_allDirections_noLight_fn)
            allVelocities_noLight = np.load(saved_allVelocities_noLight_fn)
        else: # load notes from scratch (first time analyzing or re-analyzing)
            print('Importing active movements for this experiment: '+flyExpt)
            allActiveTraces_lightOn, allDirections_lightOn, allVelocities_lightOn = plot_active_movement_traces_inactivation(expt=flyExpt, cameraView='frontal', plotTracesSeparately = 0, plotInactivate=1, savefig=savefig);
            allActiveTraces_noLight, allDirections_noLight, allVelocities_noLight = plot_active_movement_traces_inactivation(expt=flyExpt, cameraView='frontal', plotTracesSeparately = 0, plotInactivate=2, savefig=savefig);
            np.save(saved_allActiveTraces_lightOn_fn, allActiveTraces_lightOn)
            np.save(saved_allDirections_lightOn_fn, allDirections_lightOn)
            np.save(saved_allVelocities_lightOn_fn, allVelocities_lightOn)
            np.save(saved_allActiveTraces_noLight_fn, allActiveTraces_noLight)
            np.save(saved_allDirections_noLight_fn, allDirections_noLight)
            np.save(saved_allVelocities_noLight_fn, allVelocities_noLight)

        if 'allTraces_lightOn' not in locals(): #first instance of these traces (initialize with first fly's data)
            allTraces_lightOn = allActiveTraces_lightOn
            allDirs_lightOn = allDirections_lightOn
            allVels_lightOn = allVelocities_lightOn
            allTraces_noLight = allActiveTraces_noLight
            allDirs_noLight = allDirections_noLight
            allVels_noLight = allVelocities_noLight
        else: #tally up traces across all flies
            allTraces_lightOn = np.vstack((allTraces_lightOn, allActiveTraces_lightOn))
            allDirs_lightOn = np.vstack((allDirs_lightOn, allDirections_lightOn))
            allVels_lightOn = np.vstack((allVels_lightOn, allVelocities_lightOn))

            allTraces_noLight = np.vstack((allTraces_noLight, allActiveTraces_noLight))
            allDirs_noLight = np.vstack((allDirs_noLight, allDirections_noLight))
            allVels_noLight = np.vstack((allVels_noLight, allVelocities_noLight))

        # compute autocorrelation for each fly here:
        autoCorrThisFly_lightOn = np.ones((np.shape(allActiveTraces_lightOn)[0],np.shape(antInds)[0],np.shape(dirInds)[0],maxlag*2+1))
        autoCorrThisFly_lightOn[:] = np.nan
        autoCorrThisFly_noLight = np.ones((np.shape(allActiveTraces_lightOn)[0],np.shape(antInds)[0],np.shape(dirInds)[0],maxlag*2+1))
        autoCorrThisFly_noLight[:] = np.nan

        # go through every light ON trial for this fly and compute autoCorr
        for traceNum in range(np.shape(allActiveTraces_lightOn)[0]):
            for antIdx, antNum in enumerate(antInds):
                dirIdx = allDirections_lightOn[traceNum, antIdx]
                velIdx = allVelocities_lightOn[traceNum, antIdx]
                if ~np.isnan(dirIdx):# & (velIdx == 1): #HARD-CODED FOR 200cm/s
                    trace_lightOn = allActiveTraces_lightOn[traceNum, antIdx, :]
                    xx = plt.acorr(trace_lightOn-np.nanmean(trace_lightOn), normed=True, usevlines=False, maxlags=maxlag) #xx[1] contains correlation vector
                    autoCorrThisFly_lightOn[traceNum, antIdx, int(dirIdx):] = xx[1]
                    if 'autoCorrAllTraces_lightOn' not in locals():  # initialize with first correlation
                        autoCorrAllTraces_lightOn = list(xx[1])  # need to get corr in list for for concatenation in a sec
                    else:
                        autoCorrAllTraces_lightOn = np.column_stack((autoCorrAllTraces_lightOn,list(xx[1])))
        # go through every light OFF trial for this fly and compute autoCorr
        for traceNum in range(np.shape(allActiveTraces_noLight)[0]):
            for antIdx, antNum in enumerate(antInds):
                dirIdx = allDirections_noLight[traceNum, antIdx]
                velIdx = allVelocities_noLight[traceNum, antIdx]
                if ~np.isnan(dirIdx):# & (velIdx == 1): #HARD-CODED FOR 200cm/s
                    trace_noLight = allActiveTraces_noLight[traceNum, antIdx, :]
                    xx = plt.acorr(trace_noLight-np.nanmean(trace_noLight), normed=True, usevlines=False, maxlags=maxlag) #xx[1] contains correlation vector
                    autoCorrThisFly_noLight[traceNum, antIdx, int(dirIdx):] = xx[1]
                    if 'autoCorrAllTraces_noLight' not in locals():  # initialize with first correlation
                        autoCorrAllTraces_noLight = list(xx[1])  # need to get corr in list for for concatenation in a sec
                    else:
                        autoCorrAllTraces_noLight = np.column_stack((autoCorrAllTraces_noLight,list(xx[1])))
        # compute autocorr average for each fly and tally up (plot later!)
        autoCorrAllFlies_lightOn[noteInd,:,:,:] = np.nanmean(autoCorrThisFly_lightOn, 0)
        autoCorrAllFlies_noLight[noteInd,:,:,:] = np.nanmean(autoCorrThisFly_noLight, 0)

    print('From ' + str(np.shape(allExpts)[0]) + ' flies:')

    # plot all the individual fly raw traces
    fig, axTrace = plt.subplots(4,len(dirInds),facecolor=const.figColor,figsize=(10,7))
    # light OFF raw traces
    for traceNum in range(np.shape(allTraces_noLight)[0]):
        for antIdx, antNum in enumerate(antInds):
            dirIdx = allDirs_noLight[traceNum,antIdx]
            velIdx = allVels_noLight[traceNum,antIdx]
            if ~np.isnan(dirIdx) & (velIdx == 1): #HARD-CODED FOR 200cm/s
                trace = allTraces_noLight[traceNum,antIdx,:]
                axTrace[antIdx,int(dirIdx)].plot(trace,
                    color=const.colors_antAngs[antNum], alpha=const.transparencyAllActiveTraces)
    # light ON raw traces
    for traceNum in range(np.shape(allTraces_lightOn)[0]):
        for antIdx, antNum in enumerate(antInds):
            dirIdx = allDirs_lightOn[traceNum,antIdx]
            velIdx = allVels_lightOn[traceNum,antIdx]
            if ~np.isnan(dirIdx) & (velIdx == 1): #HARD-CODED FOR 200cm/s
                trace = allTraces_lightOn[traceNum,antIdx,:]
                axTrace[antIdx+2,int(dirIdx)].plot(trace,
                    color=const.colors_antAngs[antNum], alpha=const.transparencyAllActiveTraces)
    # plot average trace (optional)
    allDirs_lightOn = np.squeeze(allDirs_lightOn)
    if PLOT_AVG:
        for antIdx, antNum in enumerate(antInds):
            for dirIdx, dir in enumerate(dirInds):
                for velIdx, vel in enumerate(velInds):
                    thisDirInds = np.squeeze(np.where((allDirs_lightOn==dirIdx) & (allVels_lightOn==velIdx)))
                    average = np.nanmean(allTraces_lightOn[thisDirInds,:],0)
                    stddev = np.nanstd(allTraces_lightOn[thisDirInds,:],0)
                    axTrace[antIdx,int(dirIdx)].plot(average,
                        color=const.colors_antAngs[antNum],linewidth=3)
                    if PLOT_STD:
                        axTrace[antIdx,int(dirIdx)].plot(stddev,
                            color='green', linewidth=3)
    # set background color, labels for figures (rather HARD-CODED):
    yaxis_min = -25
    yaxis_max = 25
    scaleBarSize = 10
    scaleWidth = 2
    scaleX = 10
    scaleY = -20
    for ii in [0,1,2,3]:
        for jj, dir in enumerate(dirInds):
            axTrace[ii,jj].set_facecolor(const.figColor)
            axTrace[ii,jj].set_ylim(yaxis_min, yaxis_max)
            axTrace[ii,jj].axis('off')
            if (jj == 0) & (ii == 0):
                # add scale bar (angle, in Y)
                axTrace[ii,jj].add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
                    facecolor = const.axisColor))
                axTrace[ii,jj].text(scaleX+scaleWidth*2, scaleY+scaleBarSize/2,
                    str(scaleBarSize) + const.degree_sign,color=const.axisColor,
                    fontsize=const.fontSize_angPair,horizontalalignment='left',
                    verticalalignment='center')
            if ii == 0:
                axTrace[ii,jj].set_title(str(const.windDirNames[int(dir)]),color=const.axisColor)

    # plot all the individual fly raw traces *combining all directions and left/right antenna*
    figAllTrace, axAllTrace = plt.subplots(1,2,facecolor=const.figColor,figsize=(6,5))
    # light OFF raw traces
    for traceNum in range(np.shape(allTraces_noLight)[0]):
        for antIdx, antNum in enumerate(antInds):
            dirIdx = allDirs_noLight[traceNum, antIdx]
            velIdx = allVels_noLight[traceNum, antIdx]
            if ~np.isnan(dirIdx) & (velIdx == 1): #HARD-CODED FOR 200cm/s
                trace = allTraces_noLight[traceNum, antIdx, :]
                axAllTrace[0].plot(trace,
                    color=const.medLight_gray, alpha=const.transparencyAllActiveTraces)
    # light ON raw traces
    for traceNum in range(np.shape(allTraces_lightOn)[0]):
        for antIdx, antNum in enumerate(antInds):
            dirIdx = allDirs_lightOn[traceNum, antIdx]
            velIdx = allVels_lightOn[traceNum, antIdx]
            if ~np.isnan(dirIdx) & (velIdx == 1): #HARD-CODED FOR 200cm/s
                trace = allTraces_lightOn[traceNum, antIdx, :]
                axAllTrace[1].plot(trace,
                    color=const.medLight_gray, alpha=const.transparencyAllActiveTraces)
    # set background color, labels for figures (rather HARD-CODED):
    yaxis_min = -40
    yaxis_max = 40
    scaleBarSize = 10
    scaleWidth = 0.5
    scaleX = 10
    scaleY = -25
    for ii in [0,1]:
        axAllTrace[ii].set_facecolor(const.figColor)
        axAllTrace[ii].set_ylim(yaxis_min, yaxis_max)
        axAllTrace[ii].axis('off')
        if (ii == 1):
            # add scale bar (angle, in Y)
            axAllTrace[ii].add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
                facecolor = const.axisColor))
            axAllTrace[ii].text(scaleX+scaleWidth*2, scaleY+scaleBarSize/2,
                str(scaleBarSize) + const.degree_sign,color=const.axisColor,
                fontsize=const.fontSize_angPair,horizontalalignment='left',
                verticalalignment='center')
    axAllTrace[0].set_title('no light',color=const.axisColor)
    axAllTrace[1].set_title('light on',color=const.axisColor)


    # Plot auto correlogram for all flies (and plot average) NO LIGHT
    figAC, axTraceAC_noLight = plt.subplots(np.shape(antInds)[0],len(dirInds),figsize=(10,7),facecolor=const.figColor)
    figAC.suptitle('Autocorrelation average for single (green) and across flies (gray)\n no light',color=const.axisColor)#, nlags='+str(maxlag))
    for ai in range(np.shape(antInds)[0]):
        # plot each fly's average autocorr
        for ii, dir in enumerate(dirInds):
            for flyNum in range(np.shape(autoCorrAllFlies_noLight)[0]):
                axTraceAC_noLight[ai,ii].plot(autoCorrAllFlies_noLight[flyNum,ai,ii,:],color='green')
                halfMaxWid = np.squeeze(np.where(autoCorrAllFlies_noLight[flyNum,ai,ii,:]>0.5))
                if np.shape(halfMaxWid):
                    if np.size(halfMaxWid)>1:
                        halfMaxWid = halfMaxWid[-1]/const.framerate
                    else:
                        halfMaxWid = halfMaxWid/const.framerate
                    #print('Half max width (s) ' + str(halfMaxWid))
            #plot average autocorr across flies
            crossFlyMeanAutocorr_noLight = np.nanmean(autoCorrAllFlies_noLight[:,ai,ii,:],0)
            axTraceAC_noLight[ai,ii].plot(crossFlyMeanAutocorr_noLight,color='gray')
            set_axis_prefs_autocorr(axTraceAC_noLight[ai,ii], maxlag) #set axis prefs
            axTraceAC_noLight[ai,ii].set_title(str(const.windDirNames[int(dir)]),color=const.axisColor)
            if ii == 0:
                axTraceAC_noLight[ai,ii].set_ylabel('autocorrelation'+const.angPairNames[cameraView][antInds[ai]],color=const.axisColor)

    # Plot auto correlogram for all flies LIGHT ON (and plot average)
    figAC, axTraceAC = plt.subplots(np.shape(antInds)[0],len(dirInds),figsize=(10,7),facecolor=const.figColor)
    figAC.suptitle('Autocorrelation average for single (green) and across flies (gray) \n light activation',color=const.axisColor)#, nlags='+str(maxlag))
    for ai in range(np.shape(antInds)[0]):
        # plot each fly's average autocorr
        for ii, dir in enumerate(dirInds):
            for flyNum in range(np.shape(autoCorrAllFlies_lightOn)[0]):
                axTraceAC[ai,ii].plot(autoCorrAllFlies_lightOn[flyNum,ai,ii,:],color='green')
                halfMaxWid = np.squeeze(np.where(autoCorrAllFlies_lightOn[flyNum,ai,ii,:]>0.5))
                if np.shape(halfMaxWid):
                    if np.size(halfMaxWid)>1:
                        halfMaxWid = halfMaxWid[-1]/const.framerate
                    else:
                        halfMaxWid = halfMaxWid/const.framerate
                    #print('Half max width (s) ' + str(halfMaxWid))
            #plot average autocorr across flies
            crossFlyMeanAutocorr_lightOn = np.nanmean(autoCorrAllFlies_lightOn[:,ai,ii,:],0)
            axTraceAC[ai,ii].plot(crossFlyMeanAutocorr_lightOn,color='gray')
            set_axis_prefs_autocorr(axTraceAC[ai,ii], maxlag) #set axis prefs
            axTraceAC[ai,ii].set_title(str(const.windDirNames[int(dir)]),color=const.axisColor)
            if ii == 0:
                axTraceAC[ai,ii].set_ylabel('autocorrelation'+const.angPairNames[cameraView][antInds[ai]],color=const.axisColor)

    # Plot auto correlogram for all flies: no light vs. light overlaid
    figAllAutoCorr, axAllAuroCorr = plt.subplots(1,1,figsize=(6,6),facecolor=const.figColor)
    numNoLightTraces = np.shape(autoCorrAllTraces_noLight)[1]
    numlightTraces = np.shape(autoCorrAllTraces_lightOn)[1]
    print(str(numNoLightTraces)+' no light active movements')
    print(str(numlightTraces)+' with light (inactivation)')
    axAllAuroCorr.set_title('Autocorrelation average for active movement traces\n no light (gray, '
            +str(numNoLightTraces)+' traces) \nvs. light inactivation (green, '+str(numlightTraces)+' traces)\n',
            color=const.axisColor,fontsize=10)

    axAllAuroCorr.plot(np.nanmean(autoCorrAllTraces_noLight,axis=1),color='gray')
    axAllAuroCorr.plot(np.nanmean(autoCorrAllTraces_lightOn,axis=1),color='green')
    set_axis_prefs_autocorr(axAllAuroCorr, maxlag) #set axis prefs

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        if PLOT_STD:
            extraText = 'withStd'
        elif PLOT_AVG:
            extraText = 'withAvg'
        else:
            extraText = 'noAvg'
        savepath = figPath + 'rawActiveTracesAllFliesInactivation_' + expt +'_'+ extraText+ '.png'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + 'rawActiveTracesAllFliesInactivation_' + expt + '_'+extraText+ '.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        # save autocorrelation plot with single-fly averages
        savepath = figPath + 'activeTraces_autocorr_inactivate_' + expt + '.png'
        figAC.savefig(savepath, facecolor=figAC.get_facecolor())
        savepath = figPath + 'activeTraces_autocorr_inactivate_' + expt + '.pdf'
        figAC.savefig(savepath, facecolor=figAC.get_facecolor())
        # save average autocorrelation comparison plot (ipsi vs. contra, wind vs. odor)

        savepath = figPath + 'autocorr_avg_wind_inactivate_' + expt + '.png'
        figAllAutoCorr.savefig(savepath, facecolor=figAllAutoCorr.get_facecolor())
        savepath = figPath + 'autocorr_avg_wind_inactivate_' + expt + '.pdf'
        figAllAutoCorr.savefig(savepath, facecolor=figAllAutoCorr.get_facecolor())

        savepath = figPath + 'allRawActiveTraces_' + expt + '.png'
        figAllTrace.savefig(savepath, facecolor=figAllTrace.get_facecolor())
        savepath = figPath + 'allRawActiveTraces_' + expt + '.pdf'
        figAllTrace.savefig(savepath, facecolor=figAllTrace.get_facecolor())

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()


# Plot all raw traces for a single fly, across single direction and all velocities
# if saveFig=1, will save the figure (like many of these plotting functions)
def plot_raw_traces_odor(expt='2021_09_09_E3', cameraView='frontal', savefig=0, isTransparent=False):
    TESTING = 0 # if 1, will plot average wind on top of traces
    [angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight] = getAntennaTracesAvgs_singleExpt(expt, cameraView, 0)
    #this gives us much more than we need (tracesAvg) but okay!
    [onsetAvg, onsetAllFlies, offsetAvg, offsetAllFlies, baseVals, tracesAvg, tracesAllIndv, baseAllFliesFullTrace] = get_single_expt_lightResponse(expt, cameraView, 0)
    exptNotes = get_mat_notes(expt)
    font = FontProperties() #set font
    font.set_family(const.fontFamily)
    font.set_name(const.fontName)

    fig, axAng = plt.subplots(facecolor=[0,0,0],figsize=(6,12))

    valveStates = exptNotes.valveState[:]
    windDir = valveStates.str.get(0)/2-1 #convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)

    uniqueValveStates = np.unique(valveStates.str.get(0))
    numPreInds = int(exptNotes.pre_trial_time[0]*const.framerate)
    baselineInds = list(range(0, numPreInds))
    ncols = 1
    nrows = 5

    yaxis_max = 50
    yaxis_min = -20
    scaleBarSize = 10
    scaleWidth = 3
    scaleX = const.lenVideo*const.framerate+const.framerate/4 #put vertical scale bar to the right of the traces
    scaleY = 5
    stimStart = const.activateAvgSt

    fig.suptitle(exptNotes.notes[0] + ' ' + exptNotes.date[0] + '_E' + str(exptNotes.expNumber[0][0]), fontsize=const.fsize_title, color=const.axisColor)
    for dirIdx, state in enumerate(uniqueValveStates):
        dir = uniqueValveStates[dirIdx]
        ax = plt.subplot(nrows, ncols, dirIdx+1, frameon=False)
        # plot column titles
        if dirIdx == 0:
            plt.title('100 cm/s', fontsize=const.fsize_velTuning_sub,
                color=const.axisColor)
            ax.text(-int(stimStart*const.framerate)*0.75, 2*const.shiftYTraces,
                str('no wind'), fontsize=const.fsize_velTuning_sub,
                horizontalalignment='center', rotation=90, color=const.axisColor)
            # add scale bar (angle, in Y)
            ax.add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
                facecolor = const.axisColor))
            ax.text(scaleX+scaleWidth*2, scaleY+scaleBarSize/2,
                str(scaleBarSize) + const.degree_sign,color=const.axisColor,
                fontsize=const.fontSize_angPair,horizontalalignment='left',
                verticalalignment='center')
        else: # plot row names
            ax.text(-int(stimStart*const.framerate)*0.75, 2*const.shiftYTraces,
                str(const.windOdorNames[dir-1]) + const.degree_sign,
                fontsize=const.fsize_velTuning_sub,horizontalalignment='center',
                rotation=90, color=const.axisColor)

        inds = exptNotes.index[(exptNotes.valveState == dir) == True].tolist()
        rectX = int(const.windStart*const.framerate)
        rectWid = int(const.windTime*const.framerate)
        rectY_wind = yaxis_min+1.5
        if (dirIdx == 0):
            ax.text(rectX+rectWid/2, rectY_wind+const.stimBar_height*2+const.fontSize_stimBar/const.scaleAng_rawTraces,
                str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,
                horizontalalignment='center',color=const.axisColor)
        ax.add_patch(Rectangle((rectX,rectY_wind),rectWid,const.stimBar_height,
        facecolor = const.axisColor))

        ax.set_ylim(yaxis_min, yaxis_max)
        ax.axis('off')
        for ang in range(angs_all.shape[2]):#[1,3]:
            shift = ang*const.shiftYTraces  # shifts traces relative to each other (ease of viewing)
            for exptNum in inds:
                baseline = avgBase[exptNum][ang]
                plt.plot(angs_all[exptNum, :, ang]-baseline+shift, linewidth=const.traceWidRaw,
                    color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                if TESTING:
                    plt.plot(const.windStart*const.framerate, avgWind[exptNum][ang]-baseline+shift, color='black', marker='*')
            thisDirTraces = angs_all[inds, :, ang]
            avgTrace = np.nanmean(thisDirTraces,0)
            baseline = np.nanmean(avgTrace[baselineInds])
            plt.plot(avgTrace-baseline+shift, linewidth=const.traceWidAvg,
                color=const.colors_antAngs[ang])# alpha=const.transparencyAntTrace)
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath + 'traces_' + expt + '.png'
        print('Saving figure here: ' + savepath)
        if isTransparent:
            plt.savefig(savepath, transparent=isTransparent)
        else:
            plt.savefig(savepath, facecolor=fig.get_facecolor())

# using shutter on signal (analog signal that controls the shutter)
# return trials where light was on (activation/inactivation)
def get_activation_trials(expt='2021_11_12_E3'):
    exptNotes = get_mat_notes(expt)
    numTrials = np.shape(exptNotes)[0]
    import loadNotes_mnOptoVelocityTuning as ln
    exptType = get_expt_type(expt)
    activate = np.zeros(np.shape(exptNotes.valveState),dtype=np.int8)
    if exptType in ln.activation_note_names:
        for trialNum in range(np.shape(exptNotes)[0]):
            shutter = exptNotes.shutterOnSig[trialNum]
            activate[trialNum] = int(np.sum(shutter)>0)

    return activate

# Plot all raw traces for a single fly, across single direction and all velocities
def plot_single_expt_traces(expt='2020_11_13_E4', cameraView='frontal', importAnew = 0, plotInactivate=0, savefig=0):
    TESTING = 0  # if 1, will plot average wind on top of traces
    allowFlight = 1
    exptNotes = get_mat_notes(expt)

    [angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight] = getAntennaTracesAvgs_singleExpt(expt, cameraView, 0, allowFlight,plotInactivate=plotInactivate)
    angs_all = detect_tracking_errors_raw_trace(exptNotes, expt, angs_all)
    exptNotes = get_mat_notes(expt)
    activate = get_activation_trials(expt)

    if np.sum(activate) != np.shape(activate):
        lightCompare = 1 #will compare trials with and without light (e.g. for inactivation experiment)
    isodor = is_odor(expt)  # if odor experiments, will plot odor and no odor separately

    font = FontProperties()  # set font
    font.set_family(const.fontFamily)
    font.set_name(const.fontName)
    fig, axAng = plt.subplots(facecolor=const.figColor, figsize=(8, 8))

    windVel = exptNotes.velocity[:]
    windVel = windVel.str.get(0)  # take values out of brackets
    valveStates = exptNotes.valveState[:]
    windDir = valveStates.str.get(0)/2-1  # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)
    windDirs = np.unique(windDir)
    speeds = np.unique(windVel)
    uniqueValveStates = np.unique(valveStates.str.get(0))
    flights,_,_ = getFlightsSingleExpt(expt)

    if isodor:  # hard-coded
        ncols = 3
        nrows = 2
    else:
        ncols = np.size(speeds, 0)
        nrows = np.size(windDirs, 0)

    yaxis_max = 50
    yaxis_min = -10
    scaleBarSize = 10
    scaleWidth = 3
    scaleX = const.lenVideo*const.framerate+const.framerate/4 #put vertical scale bar to the right of the traces
    scaleY = 5
    stimStart = const.activateAvgSt
    baseInds = range(0,1*const.framerate) #baseline is first second of trace (pre-activation)
    fig.suptitle(exptNotes.notes[0] + ' ' + exptNotes.date[0] + '_E' + str(exptNotes.expNumber[0][0]), fontsize=const.fsize_title)
    for dirIdx, state in enumerate(uniqueValveStates):
        dir = uniqueValveStates[dirIdx]
        for velIdx, sp in enumerate(speeds):
            if isodor:
                ax = plt.subplot(nrows, ncols, dirIdx+1, frameon=False)
            else:
                ax = plt.subplot(nrows, ncols, ncols*dirIdx+velIdx+1, frameon=False)
            vel = speeds[velIdx]
            # plot column titles
            if dirIdx == 0:
                plt.title(str(vel) + ' cm/s', fontsize=const.fsize_velTuning_sub,
                    color=const.axisColor)
            # plot row names
            if velIdx == 0 & dirIdx == 0:
                ax.text(-int(stimStart*const.framerate)*0.75, 2*const.shiftYTraces,
                    str(const.windDirNames[dirIdx]) + const.degree_sign,
                    fontsize=const.fsize_velTuning_sub, horizontalalignment='center',
                    rotation=90, color=const.axisColor)
            if (velIdx == 0) & (dirIdx == 0):
                # add scale bar (angle, in Y)
                ax.add_patch(Rectangle((scaleX,scaleY), scaleWidth, scaleBarSize,
                    facecolor = const.axisColor))
                ax.text(scaleX+scaleWidth*2, scaleY+scaleBarSize/2,
                    str(scaleBarSize) + const.degree_sign, color=const.axisColor,
                    fontsize=const.fontSize_angPair, horizontalalignment='left',
                    verticalalignment='center')
            inds = exptNotes.index[(exptNotes.velocity == vel) & (exptNotes.valveState == dir) == True].tolist()

            # draw wind and light activation stimulus bar
            if plotInactivate != 2: #don't draw for inactivation experiments when light is off
                rectX = int(const.activateStart*const.framerate)
                rectY = yaxis_min
                rectWid = int(const.activateTime*const.framerate)
                ax.add_patch(Rectangle((rectX,rectY),rectWid,const.stimBar_height,facecolor = const.color_activateColor))

            rectX = int(const.windStart*const.framerate)
            rectWid = int(const.windTime*const.framerate)
            rectY_wind = yaxis_min+1.5
            if (velIdx == 0) & (dirIdx == 0):
                if plotInactivate != 2:
                    ax.text(rectX+rectWid, rectY-const.stimBar_height*2-const.fontSize_stimBar/const.scaleAng_rawTraces,
                        str(const.activateTime)+' s light on',color=const.color_activateColor,fontsize=const.fontSize_stimBar,horizontalalignment='left')
                ax.text(rectX+rectWid, rectY_wind+const.stimBar_height+const.fontSize_stimBar/const.scaleAng_rawTraces,
                    str(const.windTime)+' s wind', fontsize=const.fontSize_stimBar,
                    horizontalalignment='left', color=const.axisColor)
            ax.add_patch(Rectangle((rectX,rectY_wind), rectWid,const.stimBar_height,
                         facecolor = const.axisColor))

            ax.set_ylim(yaxis_min, yaxis_max)
            ax.axis('off')
            # plot raw angles
            for ang in range(angs_all.shape[2]):
                for exptNum in inds:

                    flight = flights[exptNum]
                    baseline = np.nanmean(angs_all[exptNum, baseInds, ang])
                    shift = ang*const.shiftYTraces  # shifts traces relative to each other (ease of viewing)
                    if ((plotInactivate==2) & (activate[exptNum]==0)) | (plotInactivate==0) | ((plotInactivate==1) & (activate[exptNum]==1)):
                        plt.plot(angs_all[exptNum, :, ang]-baseline+shift,
                            color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                    if TESTING:
                        plt.plot(const.windStart*const.framerate, avgWind[exptNum][ang]-baseline+shift, color='black', marker='*')

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        if plotInactivate==1:
            pltInd = 'lightOn'
        elif plotInactivate==2:
            pltInd = 'nolight'
        else: pltInd = ''
        savepath = figPath + exptNotes.notes[0] +'traces_' + expt + '_' + pltInd + '.png'
        print('Saving figure here: ' + savepath)
        fig.savefig(savepath, facecolor=fig.get_facecolor())
        savepath = figPath + exptNotes.notes[0] +'traces_' + expt+ '_' + pltInd + '.pdf'
        fig.savefig(savepath, facecolor=fig.get_facecolor())


# Plot all raw traces for a single fly, across directions and velocities
def plot_single_expt_traces_one_dir(expt='2020_11_13_E4', cameraView='frontal',savefig=0):
    dirIdx = 2
    TESTING = 0 # if 1, will plot average wind on top of traces
    [angs_all, avgBase, avgLight, avgWind, avgBaseFlight, avgLightFlight, avgWindFlight] = getAntennaTracesAvgs_singleExpt(expt, cameraView, 0)
    exptNotes = get_mat_notes(expt)
    font = FontProperties() #set font
    font.set_family(const.fontFamily)
    font.set_name(const.fontName)
    fig, axAng = plt.subplots(facecolor=[1,1,1],figsize=(8,5))

    windVel = exptNotes.velocity[:]
    windVel = windVel.str.get(0) #take values out of brackets
    valveStates = exptNotes.valveState[:]
    windDir = valveStates.str.get(0)/2-1 #convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)
    windDirs = np.unique(windDir)
    speeds = np.unique(windVel)
    uniqueValveStates = np.unique(valveStates.str.get(0))
    ncols = np.size(speeds, 0)
    nrows = 1

    yaxis_max = 50
    yaxis_min = -10
    scaleBarSize = 10
    scaleWidth = 3
    scaleX = const.lenVideo*const.framerate+const.framerate/4 #put vertical scale bar to the right of the traces
    scaleY = 5
    stimStart = const.activateAvgSt

    fig.suptitle(exptNotes.notes[0] + ' ' + exptNotes.date[0] + '_E' + str(exptNotes.expNumber[0][0]), fontsize=const.fsize_title-2)
    for velIdx, sp in enumerate(speeds):
        print(velIdx)
        ax = plt.subplot(nrows, ncols, velIdx+1, frameon=False)
        vel = speeds[velIdx]
        # plot column titles
        plt.title(str(vel) + ' cm/s', fontsize=const.fsize_velTuning_sub,
        color=const.axisColor)
        # plot row names
        if velIdx == 0:
            ax.text(-int(stimStart*const.framerate)*0.75, 2*const.shiftYTraces,
                str(const.windDirNames[dirIdx]) + const.degree_sign,
                fontsize=const.fsize_velTuning_sub,horizontalalignment='center',
                rotation=90, color=const.axisColor)
        if (velIdx == 0):
        # add scale bar (angle, in Y)
            ax.add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
                facecolor = const.axisColor))
            ax.text(scaleX+scaleWidth*2, scaleY+scaleBarSize/2,
                str(scaleBarSize) + const.degree_sign,color=const.axisColor,
                fontsize=const.fontSize_angPair,horizontalalignment='left',
                verticalalignment='center')
        inds = exptNotes.index[(exptNotes.velocity == vel) & (exptNotes.valveState == dirIdx) == True].tolist()

        #draw wind and light activation stimulus bar
        rectX = int(const.activateStart*const.framerate)
        rectY = yaxis_min#-rectHeight*.05
        rectWid = int(const.activateTime*const.framerate)
        ax.add_patch(Rectangle((rectX,rectY),rectWid,const.stimBar_height,facecolor = const.color_activateColor))

        rectX = int(const.windStart*const.framerate)
        rectWid = int(const.windTime*const.framerate)
        rectY_wind = yaxis_min+1.5
        if (velIdx == 0):
            ax.text(rectX+rectWid, rectY-const.stimBar_height*2-const.fontSize_stimBar/const.scaleAng_rawTraces,
                str(const.activateTime)+' s light on',color=const.color_activateColor,fontsize=const.fontSize_stimBar,horizontalalignment='left')
            ax.text(rectX+rectWid, rectY_wind+const.stimBar_height+const.fontSize_stimBar/const.scaleAng_rawTraces,
                str(const.windTime)+' s wind',fontsize=const.fontSize_stimBar,
                horizontalalignment='left',color=const.axisColor)
        ax.add_patch(Rectangle((rectX,rectY_wind),rectWid,const.stimBar_height,
            facecolor = const.axisColor))

        ax.set_ylim(yaxis_min, yaxis_max)
        ax.axis('off')
        for ang in range(angs_all.shape[2]):
            for exptNum in inds:
                baseline = avgBase[exptNum][ang]
                shift = ang*const.shiftYTraces  # shifts traces relative to each other (ease of viewing)
                plt.plot(angs_all[exptNum, :, ang]-baseline+shift,
                    color=const.colors_antAngs[ang], alpha=const.transparencyAntTrace)
                if TESTING:
                    plt.plot(const.windStart*const.framerate, avgWind[exptNum][ang]-baseline+shift, color='black', marker='*')

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath + 'traces_' + expt + '.png'
        print('Saving figure here: ' + savepath)
        plt.savefig(savepath, transparent=True)


# hard-coded flip of left second segment traces onto right
# if thirdSeg = 1, will return combined third segment traces (default is 2nd)
def flipLeftTraces(tracesAllFlies, thirdSeg = 0):
    if thirdSeg == 1:
        rightTraces = tracesAllFlies[:,:,1,:,:] # right 3rd segment traces
        leftTraces = tracesAllFlies[:,:,3,:,:]  # left 3rd segment traces
    else:
        rightTraces = tracesAllFlies[:,:,0,:,:] # right 2nd segment traces
        leftTraces = tracesAllFlies[:,:,2,:,:] # right 2nd segment traces
    # mirror left traces around zero direction
    leftSecondMirrored = np.empty(np.shape(leftTraces))
    leftSecondMirrored[:,:,0,:] = leftTraces[:,:,4,:]
    leftSecondMirrored[:,:,1,:] = leftTraces[:,:,3,:]
    leftSecondMirrored[:,:,2,:] = leftTraces[:,:,2,:] # frontal is frontal, no flipping
    leftSecondMirrored[:,:,3,:] = leftTraces[:,:,1,:]
    leftSecondMirrored[:,:,4,:] = leftTraces[:,:,0,:]
    # combine the traces into one structure!
    combinedTraces = np.concatenate((rightTraces,leftSecondMirrored),axis=0)

    return combinedTraces

# Plot cross-fly antenna angles for a given experiment type
# input: singleDir = -1 --> plot all directions, antenna angles, velocities
#        singleDir = 0,1,2,3,4 --> plot single directions (-90, -45, 0, +45, +90)
#        singleDir = -2 --> plot all velocities, directions, for just the second segment (and flip left onto right antenna)
# avgOnly --> if 1, will plot just the average (no single fly traces)
def plot_cross_fly_traces(expt='CS_activate',cameraView='frontal',singleDir=1, avgOnly = 0, allowFlight=0,plotInactivate = 0,savefig=0):
    #avgOnly = 1  # if 1, will plot average across traces only (no single flies)
    PLOT_3_MIN_2 = 1 # if 1, will plot third segment minus second and second alone (rather than third and second)
    import loadNotes_mnOptoVelocityTuning as ln
    thirdSeg = 0 #set this to one if we want to check out third segment movements
    allExpts = eval('ln.notes_'+expt)
    tracesAllFlies = getAntennaTracesAvgs_crossExpt(expt, cameraView, 0,allowFlight=allowFlight,plotInactivate=plotInactivate)

    if singleDir == -2: #plot just the second segment, and flip left onto right
        tracesAllFlies = flipLeftTraces(tracesAllFlies,thirdSeg)
        anglesToPlot = [0] #will plot the single set of 2nd segment traces
        shiftAngY = 0
        if avgOnly == 1:
            yaxis_max = 10
            ylimAll = [-2,8]
            scaleY = 1
            yaxis_min = -2
        else:
            yaxis_max = 20
            ylimAll = [-5, 30]
            scaleY = 5
            yaxis_min = -5
        textYPos = 1
    else:
        anglesToPlot = range(tracesAllFlies.shape[2])
        shiftAngY = const.shiftYTracesCrossFly
        yaxis_max = 50
        ylimAll = [-5, 80]
        textYPos = 2*const.shiftYTraces
        scaleY = 5
        yaxis_min = -5
    if avgOnly == 1:
        scaleBarSize = 3
        stimBarHeight = 0.25
        rectY = yaxis_min+0.1
        rectYWind = yaxis_min+0.7
    else:
        scaleBarSize = 5
        stimBarHeight = const.stimBar_height
        rectY = yaxis_min+0.1
        rectYWind = yaxis_min+1.5
    scaleWidth = 10
    scaleX = const.lenVideo*const.framerate+const.framerate/4 #put vertical scale bar to the right of the traces
    stimStart = const.activateAvgSt

    font = FontProperties()  # set font
    font.set_family(const.fontFamily)
    font.set_name(const.fontName)

    exptNotes = get_mat_notes(allExpts[0]['date'])
    windVel = exptNotes.velocity[:]
    windVel = windVel.str.get(0)  # take values out of brackets
    valveStates = exptNotes.valveState[:]
    windDir = valveStates.str.get(0)/2-1  # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)
    windDirs = np.unique(windDir)
    speeds = np.unique(windVel)
    uniqueValveStates = np.unique(valveStates.str.get(0))
    ncols = np.size(speeds, 0)
    nrows = np.size(uniqueValveStates, 0)
    numPreInds = int(exptNotes.pre_trial_time[0]*const.framerate)
    baselineInds = list(range(0, numPreInds))

    # scale bar constants for x-sizing and positioning (y above depends on plotting prefs)
    rectX = int(const.activateStart*const.framerate)
    rectWid = int(const.activateTime*const.framerate)
    rectXWind = int(const.windStart*const.framerate)
    rectWidWind = int(const.windTime*const.framerate)

    if singleDir == 0: rectHeight = const.stimBar_height
    else: rectHeight = const.stimBar_height*0.5

    if singleDir < 0:
        fig, axAng = plt.subplots(nrows, ncols, facecolor=const.figColor, figsize=(8, 8))
    else:
        fig, axAng = plt.subplots(1, ncols, facecolor=const.figColor, figsize=(8, 6))
        uniqueValveStates = [singleDir]# hard-coded: plot for -45 deg
    # plot single-fly traces for all velocities and directions
    for dirIdx, state in enumerate(uniqueValveStates):
        for velIdx, sp in enumerate(speeds):
            if singleDir < 0:
                axIdx = dirIdx, velIdx
            else: axIdx = velIdx
            # plot column titles
            if dirIdx == 0:
                if velIdx == 0: #also include N=#flies here
                    axAng[axIdx].set_title(str(sp) + ' cm/s N='+str(np.shape(tracesAllFlies)[0])+' flies', fontsize=const.fsize_velTuning_sub,
                        color=const.axisColor)
                else:
                    axAng[axIdx].set_title(str(sp) + ' cm/s', fontsize=const.fsize_velTuning_sub,
                        color=const.axisColor)
            # plot row names
            if (velIdx == 0):
                if singleDir <0:
                    yText = str(const.windDirLabelsIC[dirIdx]) + const.degree_sign
                else:
                    yText = str(const.windDirNames[uniqueValveStates[dirIdx]]) + const.degree_sign
                axAng[axIdx].text(-int(stimStart*const.framerate)*0.75, textYPos, yText,
                    fontsize=const.fsize_velTuning_sub, horizontalalignment='center',
                    rotation=90, color=const.axisColor)
            if (velIdx == 0) & (dirIdx == 0):
                # add scale bar (angle, in Y)
                axAng[axIdx].add_patch(Rectangle((scaleX,scaleY), scaleWidth, scaleBarSize,
                    facecolor = const.axisColor))
                axAng[axIdx].text(scaleX+scaleWidth*2, scaleY+scaleBarSize/2,
                    str(scaleBarSize) + const.degree_sign, color=const.axisColor,
                    fontsize=const.fontSize_angPair, horizontalalignment='left',
                    verticalalignment='center')
            # draw wind and light activation stimulus bar
            # hard-coded: show stim bar only for bottom row when showing simpliied 2nd segment averages
            if (singleDir != -2) | ((singleDir == -2) &(avgOnly == 1) & (dirIdx == 4)):
                axAng[axIdx].add_patch(Rectangle((rectX,rectY),rectWid,
                        stimBarHeight,facecolor = const.color_activateColor))
                if ((velIdx == 0) & (dirIdx == 0)) | (singleDir == -2):
                    axAng[axIdx].text(rectX+rectWid, rectY-const.fontSize_stimBar/const.scaleAng_rawTraces,
                        str(const.activateTime)+' s light on',color=const.color_activateColor,fontsize=const.fontSize_stimBar,horizontalalignment='left')
                    axAng[axIdx].text(rectX+rectWid, rectYWind,
                        str(const.windTime)+' s wind', fontsize=const.fontSize_stimBar,
                        horizontalalignment='left', color=const.axisColor)
                axAng[axIdx].add_patch(Rectangle((rectXWind,rectYWind), rectWidWind,
                    stimBarHeight, facecolor = const.axisColor))
            # plot the traces
            for ang in anglesToPlot:
                if singleDir == -2:  # if one segment plot in grayscale
                    traceColor=const.axisColor
                    traceColorAvg = const.med_gray
                elif (PLOT_3_MIN_2 == 1) & (ang == 1):
                    traceColor = const.green32
                    traceColorAvg = traceColor
                elif (PLOT_3_MIN_2 == 1) & (ang == 3):
                    traceColor = const.purple32
                    traceColorAvg = traceColor
                else:  # if multiple segments plot in antenna segment colors
                    traceColor=const.colors_antAngs[ang]
                    traceColorAvg = traceColor
                for flyNum in range(tracesAllFlies.shape[0]):
                    shift = ang*shiftAngY  # shifts traces relative to each other (ease of viewing)
                    if singleDir >= 0:
                        trace = tracesAllFlies[flyNum, :, ang, uniqueValveStates[dirIdx], velIdx]
                    elif singleDir == -2:
                        trace = tracesAllFlies[flyNum, :, dirIdx, velIdx]
                    elif (PLOT_3_MIN_2 == 1) & ((ang == 3) | (ang == 1)): #plot 3rd-2nd segments rather than third alone (i.e. 3-2 and 1-0)
                        trace = tracesAllFlies[flyNum, :, ang, dirIdx, velIdx]-tracesAllFlies[flyNum, :, ang-1, dirIdx, velIdx]
                    else:
                        trace = tracesAllFlies[flyNum, :, ang, dirIdx, velIdx]
                    baseline = np.nanmean(trace[baselineInds])
                    if avgOnly == 0:
                        axAng[axIdx].plot(trace-baseline+shift,
                            color=traceColor, alpha=const.transparencySingleFlyTrace)
                # plot cross-fly average over single-fly averages
                shift = ang*const.shiftYTracesCrossFly  # shifts traces relative to each other (ease of viewing)
                if singleDir >= 0:
                    trace = np.nanmean(tracesAllFlies[:, :, ang, uniqueValveStates[dirIdx], velIdx], axis=0)
                elif singleDir == -2:
                    trace = np.nanmean(tracesAllFlies[:, :, dirIdx, velIdx], axis=0)
                elif (PLOT_3_MIN_2 == 1) & ((ang == 3) | (ang == 1)): #plot 3rd-2nd segments rather than third alone (i.e. 3-2 and 1-0)
                    trace = np.nanmean(tracesAllFlies[:, :, ang, dirIdx, velIdx], axis=0)-np.nanmean(tracesAllFlies[:, :, ang-1, dirIdx, velIdx], axis=0)
                else:
                    trace = np.nanmean(tracesAllFlies[:, :, ang, dirIdx, velIdx], axis=0)
                baseline = np.nanmean(trace[baselineInds])
                axAng[axIdx].plot(trace-baseline+shift,
                    color=traceColorAvg, alpha=const.transparencyCrossFlyTrace)

            # configure the axes
            axAng[axIdx].set_facecolor(const.figColor)
            axAng[axIdx].set_ylim(ylimAll)
            axAng[axIdx].axis('off')

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if plotInactivate ==1:
        extraText = '_withLight'
    elif plotInactivate == 2:
        extraText = '_noLight'
    else:
        extraText = ''
    if allowFlight == 1:
        extraText2 = '_withFlight'
    else:
        extraText2 = '_noFlight'

    if savefig:  # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        savepath_png = figPath + 'tracesAcrossFlies_oneDir_' +str(singleDir)+'_'+str(avgOnly)+'_'+ expt + extraText+extraText2+'.png'
        savepath_pdf = figPath + 'tracesAcrossFlies_oneDir_' +str(singleDir)+'_'+str(avgOnly)+'_'+ expt + extraText+extraText2+'.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())

def quantify_second_displacement(expt='CS_activate',cameraView='frontal',savefig=0):
    speed = 3 # 200 cm/s
    dirIdx = 4 # -45 deg
    import loadNotes_mnOptoVelocityTuning as ln
    allExpts = eval('ln.notes_'+expt)
    exptNotes = get_mat_notes(allExpts[0]['date'])
    numPreInds = int(exptNotes.pre_trial_time[0]*const.framerate)
    baselineInds = list(range(0, numPreInds))
    stopWind = int(const.windStop*const.framerate)
    startAvg = int(stopWind-(stopWind-const.windStart*const.framerate)/2)
    steadyStateStimInds = list(range(startAvg, stopWind)) # second half of stimulus region

    yaxis_max = 50
    yaxis_min = -5
    scaleBarSize = 10
    scaleWidth = 3
    scaleX = const.lenVideo*const.framerate+const.framerate/4 #put vertical scale bar to the right of the traces
    scaleY = 5
    stimStart = const.activateAvgSt
    ylimAll = [-5,70]
    # scale bar constants
    rectX = int(const.activateStart*const.framerate)
    rectY = yaxis_min+0.1
    rectWid = int(const.activateTime*const.framerate)
    rectXWind = int(const.windStart*const.framerate)
    rectWidWind = int(const.windTime*const.framerate)
    rectYWind = yaxis_min+1.5
    rectHeight = const.stimBar_height*0.5

    tracesAllFlies = getAntennaTracesAvgs_crossExpt(expt, cameraView, 0)
    secondSegStimAvg = np.empty([tracesAllFlies.shape[0], tracesAllFlies.shape[2]])
    fig, axAng = plt.subplots(1,2,facecolor=const.figColor, figsize=(4, 3))
    plt.title('200 cm/s, -45° '+expt)  # hard-coded
    # draw wind and light activation stimulus bar
    axAng[0].add_patch(Rectangle((rectX,rectY),rectWid,
            const.stimBar_height,facecolor = const.color_activateColor))
    axAng[0].text(rectX+rectWid, rectY-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(const.activateTime)+' s light on',color=const.color_activateColor,fontsize=const.fontSize_stimBar,horizontalalignment='left')
    axAng[0].text(rectXWind+rectWidWind, rectYWind, #+const.stimBar_height+const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(const.windTime)+' s wind', fontsize=const.fontSize_stimBar,
        horizontalalignment='left', color=const.axisColor)
    axAng[0].add_patch(Rectangle((rectXWind,rectYWind), rectWidWind,
            const.stimBar_height, facecolor = const.axisColor))
    # plot the traces
    for ang in range(tracesAllFlies.shape[2]):
        for flyNum in range(tracesAllFlies.shape[0]):
            shift = ang*const.shiftYTracesCrossFly  # shifts traces relative to each other (ease of viewing)
            trace = tracesAllFlies[flyNum, :, ang,dirIdx,speed]
            baseline = np.nanmean(trace[baselineInds])
            # compute baseline-subtracted average antennal response during steady state:
            secondSegStimAvg[flyNum,ang] = np.nanmean(trace[steadyStateStimInds])-baseline
            axAng[0].plot(trace-baseline+shift,
                color=const.colors_antAngs[ang], alpha=const.transparencySingleFlyTrace)
        # plot cross-fly average over single-fly averages
        shift = ang*const.shiftYTracesCrossFly  # shifts traces relative to each other (ease of viewing)
        trace = np.nanmean(tracesAllFlies[:, :, ang,dirIdx,speed],axis=0)
        baseline = np.nanmean(trace[baselineInds])
        axAng[0].plot(trace-baseline+shift,
            color=const.colors_antAngs[ang], alpha=const.transparencyCrossFlyTrace)

    jitterSft = 0.2
    jitter = jitterSft*np.random.rand(len(secondSegStimAvg[:,0]))-jitterSft/2 # add some random jitter to x-placement of single trial responses
    axAng[1].plot(np.ones(np.shape(secondSegStimAvg[:,0]))+jitter, secondSegStimAvg[:,0],alpha=const.transparencyTuningPoints,
        marker='.', markerSize = const.markerTuningIndv-1, linestyle='None', color=const.colors_antAngs[0])  # right second segment
    axAng[1].plot(1, np.nanmean(secondSegStimAvg[:,0]), markerSize = const.markerTuningAvg,
        marker='.', linestyle='None', color=const.colors_antAngs[0])  # right second segment
    jitter = jitterSft*np.random.rand(len(secondSegStimAvg[:,0]))-jitterSft/2 # add some random jitter to x-placement of single trial responses
    axAng[1].plot(np.ones(np.shape(secondSegStimAvg[:,2]))*2+jitter, secondSegStimAvg[:,2], alpha=const.transparencyTuningPoints,
        marker='.', markerSize = const.markerTuningIndv-1, linestyle='None', color=const.colors_antAngs[2])  # left second segment
    axAng[1].plot(2, np.nanmean(secondSegStimAvg[:,2]), markerSize = const.markerTuningAvg,
        marker='.', linestyle='None', color=const.colors_antAngs[2])  # right second segment

    # configure the axes
    axAng[0].set_facecolor(const.figColor)
    axAng[0].set_ylim(ylimAll)
    axAng[0].axis('off')

    # configure the axes
    axAng[1].set_facecolor(const.figColor)
    axAng[1].set_ylim([-3, 13])
    set_axis_standard_preferences(axAng[1])
    axAng[1].spines['left'].set_bounds(-2, 12)
    axAng[1].tick_params(axis='y',colors=const.axisColor)
    axAng[1].spines['left'].set_color(const.axisColor)
    #axAng[1].axis('off')

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig:  # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)
        # save raster (cross-fly)
        savepath_png = figPath + 'quantifySecondDisplacement_' + expt + '.png'
        savepath_pdf = figPath + 'quantifySecondDisplacement_' + expt + '.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())

    return secondSegStimAvg

# Plot quantification of second segment movements: end of wind stimulus vs. beginning
# Will plot single fly averages and single trial responses (more appropriate for this analysis)
def quant_second_disp_end_vs_start(expt='CS_activate',cameraView='frontal',savefig=0):
    plotFlyAvgs = 0
    if (expt == 'eyFLP_inactivate') | (expt == '91F02_inactivate') | (expt=='18D07_inactivate'):
        speed = 1 # 200 cm/s
        dirIdx = 1 # -45 deg
    else:
        speed = 3 # 200 cm/s
        dirIdx = 1 # -45 deg
    ymax = 1050 #distribution is much narrower for these flies
    ymin = -10
    xmin = -35
    xmax = 35

    antAngs = [0,1,2,3]
    import loadNotes_mnOptoVelocityTuning as ln
    allExpts = eval('ln.notes_'+expt)
    exptNotes = get_mat_notes(allExpts[0]['date'])
    numPreInds = int(exptNotes.pre_trial_time[0]*const.framerate/2)
    baselineInds = list(range(70,90))#110))#list(range(0, numPreInds))
    stopWind = int(const.windStop*const.framerate)
    stopWind = 235
    startAvg = 125#
    numAvgInds = int(const.framerate/2) #average over 1/2 second for onset/offset comparison
    onsetStimInds = list(range(startAvg, startAvg+numAvgInds)) # second half of stimulus region
    offsetStimInds = list(range(stopWind-numAvgInds, stopWind)) # second half of stimulus region
    stimInds = list(range(startAvg, stopWind))
    allowFlight = 1

    yaxis_max = 50
    yaxis_min = -15
    scaleBarSize = 10
    scaleWidth = 3
    scaleX = const.lenVideo*const.framerate#+const.framerate/4 #put vertical scale bar to the right of the traces
    scaleY = 55
    stimStart = const.activateAvgSt
    ylimAll = [-5,70]
    # scale bar constants
    rectX = int(const.activateStart*const.framerate)
    rectY = yaxis_min+0.1
    rectWid = int(const.activateTime*const.framerate)
    rectXWind = int(const.windStart*const.framerate)
    rectWidWind = int(const.windTime*const.framerate)
    rectYWind = yaxis_min+1.5
    rectHeight = const.stimBar_height*0.5

    tracesAllFlies = getAntennaTracesAvgs_crossExpt(expt, cameraView, 0)
    secondSegStimAvg_start = np.empty([tracesAllFlies.shape[0], tracesAllFlies.shape[2]])
    secondSegStimAvg_end = np.empty([tracesAllFlies.shape[0], tracesAllFlies.shape[2]])
    secondAvg_diff = np.empty([tracesAllFlies.shape[0], tracesAllFlies.shape[2]])
    secondAvg_diff_trials = np.empty([tracesAllFlies.shape[0],120, tracesAllFlies.shape[2]]) #number of trials hard-coded at 60
    secondAvg_diff_trials[:] = np.nan
    if plotFlyAvgs == 1:
        fig, axAng = plt.subplots(1,2,facecolor=const.figColor, figsize=(7, 10))
        plt.title('200 cm/s, -45° '+expt)  # hard-coded
    fig, axAng_single = plt.subplots(1,4,facecolor=const.figColor, figsize=(7, 10))

    # draw wind and light activation stimulus bar
    axAng_single[0].add_patch(Rectangle((rectX,rectY),rectWid,
            const.stimBar_height,facecolor = const.color_activateColor))
    axAng_single[0].text(rectX+rectWid, rectY-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(const.activateTime)+' s light on',color=const.color_activateColor,fontsize=const.fontSize_stimBar,horizontalalignment='left')
    axAng_single[0].text(rectXWind+rectWidWind, rectYWind, #+const.stimBar_height+const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(const.windTime)+' s wind', fontsize=const.fontSize_stimBar,
        horizontalalignment='left', color=const.axisColor)
    axAng_single[0].add_patch(Rectangle((rectXWind,rectYWind), rectWidWind,
            const.stimBar_height, facecolor = const.axisColor))

    # add vertical scale bar
    axAng_single[0].add_patch(Rectangle((scaleX,scaleY),scaleWidth,scaleBarSize,
        facecolor = const.axisColor))
    axAng_single[0].text(scaleX+scaleWidth*6, scaleY+scaleBarSize/2,
        str(scaleBarSize) + const.degree_sign,color=const.axisColor,
        fontsize=const.fontSize_angPair,horizontalalignment='left',
        verticalalignment='center')

    # add boxes around data where quantification is derived
    bottom = -10
    height = 140
    # plot early/onset box
    left = onsetStimInds[0]
    width = np.shape(onsetStimInds)[0]
    axAng_single[0].add_patch(Rectangle((left,bottom),width,height,
        edgecolor = const.med_gray, facecolor = 'None'))
    #plot offset/late box
    left = offsetStimInds[0]
    width = np.shape(offsetStimInds)[0]
    axAng_single[0].add_patch(Rectangle((left,bottom),width,height,
        edgecolor = const.med_gray, facecolor = 'None'))

    # plot the traces
    for flyNum, notes in enumerate(allExpts):
        flyExpt = notes['date']
        print(flyExpt)
        exptNotes = get_mat_notes(flyExpt)
        numTrials = np.shape(exptNotes)[0]
        # this is probably overkill to do this for each fly
        windVel = exptNotes.velocity[:]
        windVel = windVel.str.get(0)  # take values out of brackets
        valveStates = exptNotes.valveState[:]
        windDir = valveStates.str.get(0)/2-1  # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
        windDir = windDir.astype(int)
        windDirs = np.unique(windDir)
        speeds = np.unique(windVel)
        uniqueValveStates = np.unique(valveStates.str.get(0))
        [angs_all, _, _, _, _, _, _] = getAntennaTracesAvgs_singleExpt(flyExpt, cameraView, 0, allowFlight)
        angs_all = detect_tracking_errors_raw_trace(exptNotes, flyExpt, angs_all) #will exclude (nan) any very large tracking errors

        # plot all the single-trial angle traces for this fly
        inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (windDir == windDirs[dirIdx]) == True].tolist()

        for ang in antAngs:
            shift = ang*const.shiftYTracesCrossFly  # shifts traces relative to each other (ease of viewing)
            shiftSingles = ang*const.shiftYTracesCrossFly*2 #spacing of the different antenna angles
            for exptNum in inds:
                baseline = np.nanmean(angs_all[exptNum, baselineInds, ang])
                traceSingleTrial = angs_all[exptNum, :, ang]
                axAng_single[0].plot(traceSingleTrial-baseline+shiftSingles,
                    color=const.colors_antAngs[ang], alpha=const.transparencySingleFlyTrace)
                #compute single-trial differnces in second segment
                start = np.nanmean(traceSingleTrial[onsetStimInds])-baseline
                end = np.nanmean(traceSingleTrial[offsetStimInds])-baseline
                secondAvg_diff_trials[flyNum,exptNum,ang] = end-start

            if plotFlyAvgs:
                #plot right single fly average angles for contralateral 45 deg wind (-45)
                trace = tracesAllFlies[flyNum, :, ang,dirIdx,speed]
                baseline = np.nanmean(trace[baselineInds])
                axAng[0].plot(trace-baseline+shift,
                    color=const.colors_antAngs[ang], alpha=const.transparencySingleFlyTrace)
                # compute baseline-subtracted average antennal response during steady state:
                start = np.nanmean(trace[onsetStimInds])-baseline
                end = np.nanmean(trace[offsetStimInds])-baseline
                secondAvg_diff[flyNum,ang] = end-start

                # plot cross-fly average over single-fly averages
                shift = ang*const.shiftYTracesCrossFly  # shifts traces relative to each other (ease of viewing)
                trace = np.nanmean(tracesAllFlies[:, :, ang,dirIdx,speed],axis=0)
                baseline = np.nanmean(trace[baselineInds])
                axAng[0].plot(trace-baseline+shift,
                    color=const.colors_antAngs[ang], alpha=const.transparencyCrossFlyTrace)

    # plot distribution of antenna angles for each antennal segment separately
    # this is not efficient given the loop above but simpler to deal with!
    fig_hist, axAng_hist = plt.subplots(1,5,facecolor=const.figColor, figsize=(12, 4))
    plt.title('distribution of antenna angles\n during wind stimulus'+expt)  # hard-coded
    n_bins = 60
    antAngs = [3,1,2,0] #change the order in which we plot to look at 2nd segment more easily
    stimInds = list(range(startAvg, stopWind))
    namesIpsiContra = ['contra2', 'contra3','ipsi2', 'ipsi3']
    allTraces = {} #dict to store all the single angle data in across flies
    for ang in [0,1,2,3,4]:
        if ang > 0:
            axAng_hist[ang].spines['left'].set_visible(False)
            axAng_hist[ang].yaxis.set_visible(False)
        axAng_hist[ang].set_ylim([ymin, ymax])
        axAng_hist[ang].set_xlim([xmin, xmax])
        axAng_hist[ang].spines['bottom'].set_bounds(-30, 30)
        axAng_hist[ang].spines['left'].set_bounds(0, ymax)
        axAng_hist[ang].tick_params(direction='in')
        axAng_hist[ang].spines['top'].set_visible(False)
        axAng_hist[ang].spines['right'].set_visible(False)

    # plot the distributions of the data!
    for ang in antAngs:
        allSingleFlyTracesOneAng = []
        for flyNum, notes in enumerate(allExpts):
            flyExpt = notes['date']
            exptNotes = get_mat_notes(flyExpt)
            inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (exptNotes.valveState == windDirs[dirIdx]) == True].tolist()
            [angs_all, _, _, _, _, _, _] = getAntennaTracesAvgs_singleExpt(flyExpt, cameraView, 0, allowFlight)
            angs_all = detect_tracking_errors_raw_trace(exptNotes, flyExpt, angs_all)
            for exptNum in inds:
                baseline = np.nanmean(angs_all[exptNum, baselineInds, ang]) #inds are for 200cm/s, -45
                singleFlyTraces = angs_all[exptNum, stimInds, ang]-baseline
                axAng_single[3].plot(np.squeeze(singleFlyTraces), color=const.colors_antAngs[ang], alpha=const.transparencyCrossFlyTrace)
                allSingleFlyTracesOneAng = np.append(allSingleFlyTracesOneAng, singleFlyTraces)

        # plot distribution of antenna angles for each segment separately
        axAng_hist[4].hist(allSingleFlyTracesOneAng, bins=n_bins,
            color=const.colors_antAngs[ang], alpha=const.transparencySingleFlyTrace,
            edgecolor = const.colors_antAngs[ang])
        axAng_hist[4].hist(allSingleFlyTracesOneAng, bins=n_bins,
            edgecolor = const.colors_antAngs[ang], facecolor = 'none')
        #plot each histogram separately
        axAng_hist[ang].hist(allSingleFlyTracesOneAng, bins=n_bins,
            color=const.colors_antAngs[ang], alpha=const.transparencySingleFlyTrace,
            edgecolor = const.colors_antAngs[ang])
        meanAng = np.nanmean(allSingleFlyTracesOneAng)
        axAng_hist[4].plot([meanAng,meanAng],[0,ymax-5],
            color=const.colors_antAngs[ang], linestyle='--')
        axAng_hist[ang].hist(allSingleFlyTracesOneAng, bins=n_bins,
            edgecolor = const.colors_antAngs[ang], facecolor = 'none')
        axAng_hist[ang].set_title(namesIpsiContra[ang]+ ' avg=' + str(round_to_significant_digit(meanAng,2)))

        xx = str(ang)
        allTraces[xx] = allSingleFlyTracesOneAng

    # Test for normality using the Kolmogorov-Smirnov test
    contraSecond = allTraces['3']
    ipsiSecond = allTraces['2']
    ksTest_contraSecond = stats.kstest(contraSecond[~np.isnan(contraSecond)],'norm', alternative='two-sided')
    print(ksTest_contraSecond)
    ksTest_ipsiSecond = stats.kstest(ipsiSecond[~np.isnan(ipsiSecond)],'norm', alternative='two-sided')
    print(ksTest_ipsiSecond)
    # Compare distributions of non-normal antenna angles with Mann-Whitney U:
    result1 = stats.mannwhitneyu(ipsiSecond, contraSecond, alternative='two-sided') #difference between 2nd antennal segments
    pVals = [result1.pvalue]
    print(pVals)

    jitterSft = 0.05
    if plotFlyAvgs:
        # Plotting across fly averages in this figure!
        # plot second segment difference for contra antenna
        ang = 0
        jitter = jitterSft*np.random.rand(len(secondAvg_diff[:,ang]))-jitterSft/2 # add some random jitter to x-placement of single trial responses
        axAng[1].plot(np.ones(np.shape(secondAvg_diff[:,ang]))+jitter, secondAvg_diff[:,ang],alpha=const.transparencyTuningPoints,
            marker='.', markerSize = const.markerTuningIndv-1, linestyle='None', color=const.colors_antAngs[0])  # right second segment
        #plot average
        axAng[1].plot(1, np.nanmean(secondAvg_diff[:,ang]), markerSize = const.markerTuningAvg,
            marker='.', linestyle='None', color=const.colors_antAngs[ang])  # right second segment
        axAng[1].plot([1, 2],[secondAvg_diff[:,0], secondAvg_diff[:,2]],alpha=const.transparencyTuningPoints,
            linestyle='-', color=const.med_gray)
        # plot second segment of ipsi antenna for comparison
        ang = 2
        jitter = jitterSft*np.random.rand(len(secondAvg_diff[:,ang]))-jitterSft/2 # add some random jitter to x-placement of single trial responses
        # plot second segment difference
        axAng[1].plot(np.ones(np.shape(secondAvg_diff[:,ang]))*2+jitter, secondAvg_diff[:,ang],alpha=const.transparencyTuningPoints,
            marker='.', markerSize = const.markerTuningIndv-1, linestyle='None', color=const.colors_antAngs[ang])  # right second segment
        #plot average
        axAng[1].plot(2, np.nanmean(secondAvg_diff[:,ang]), markerSize = const.markerTuningAvg,
            marker='.', linestyle='None', color=const.colors_antAngs[ang])  # right second segment

        # plot result of ttest between groups (indicate significane level)
        result1 = stats.ttest_rel(secondAvg_diff[:,0], secondAvg_diff[:,2], nan_policy='omit')
        pVals = [result1.pvalue]
        result2 = stats.ttest_1samp(secondAvg_diff[:,0], popmean=0, nan_policy='omit')
        pVals2 = [result2.pvalue]
        result3 = stats.ttest_1samp(secondAvg_diff[:,2], popmean=0, nan_policy='omit')
        pVals3 = [result3.pvalue]
        print(pVals)
        axAng[1].text(0.95, 4, 'ipsi vs. contra: ' +str(round_to_significant_digit(result1.pvalue,2)), color='gray')
        axAng[1].text(0.95,3.75, 'ipsi ~ 0: ' +str(round_to_significant_digit(result2.pvalue,2)), color='gray')
        axAng[1].text(0.95,3.25, 'contra ~ 0: ' +str(round_to_significant_digit(result3.pvalue,2)), color='gray')
        # plot a representation of the p-value (ns or *, **, ***)
        yMax = 4.25
        sft = 0
        for ii in range(np.shape(pVals)[0]):
            yy = yMax-ii*sft
            axAng[1].plot([1, ii+2],[yy,yy], color=const.axisColor, linewidth=1)
            if pVals[ii] < 0.001:
                mkr = '***'
            elif pVals[ii] < 0.01:
                mkr = '**'
            elif pVals[ii] < 0.05:
                mkr = '*'
            else: mkr = 'ns'
            axAng[1].text(ii+1.5,yy+sft/10,mkr,color=const.axisColor,fontsize=const.fontSize_axis+1)

    # Plotting single-trial data in this figure!
    # plot second segment difference for contra antenna
    numFlies = np.shape(secondAvg_diff_trials)[0]
    numTrials = np.shape(secondAvg_diff_trials)[1]
    ang = 0
    jitter = jitterSft*np.random.rand(numFlies,numTrials)-jitterSft/2 # add some random jitter to x-placement of single trial responses
    axAng_single[2].plot(np.ones((numFlies,numTrials))+jitter, secondAvg_diff_trials[:,:,ang],alpha=const.transparencyTuningPoints,
        marker='.', markerSize = const.markerTuningIndv-1, linestyle='None', color=const.colors_antAngs[0])  # right second segment
    #plot average
    axAng_single[2].plot(1, np.nanmean(secondAvg_diff_trials[:,:,ang]), markerSize = const.markerTuningAvg,
        marker='.', linestyle='None', color=const.colors_antAngs[ang])  # right second segment

    # combine all single-trial data (across flies)
    allFlyTrials_0 = []
    allFlyTrials_2 = []
    for tt in range(numFlies):
        allFlyTrials_0 = np.append(allFlyTrials_0,np.squeeze(secondAvg_diff_trials[tt,:,0]))
        allFlyTrials_2 = np.append(allFlyTrials_2,np.squeeze(secondAvg_diff_trials[tt,:,2]))

    for tt in range(numFlies):
        axAng_single[2].plot([np.ones(120),np.ones(120)*2],[np.squeeze(secondAvg_diff_trials[tt,:,0]), np.squeeze(secondAvg_diff_trials[tt,:,2])],alpha=const.transparencyTuningPoints,
            linestyle='-',color=const.med_gray)
    # plot second segment of ipsi antenna for comparison
    ang = 2
    jitter = jitterSft*np.random.rand(numFlies,numTrials)-jitterSft/2 # add some random jitter to x-placement of single trial responses
    # plot second segment difference
    axAng_single[2].plot(np.ones((numFlies,numTrials))*2+jitter, secondAvg_diff_trials[:,:,ang],alpha=const.transparencyTuningPoints,
        marker='.', markerSize = const.markerTuningIndv-1, linestyle='None', color=const.colors_antAngs[ang])  # right second segment
    #plot average
    axAng_single[2].plot(2, np.nanmean(secondAvg_diff_trials[:,:,ang]), markerSize = const.markerTuningAvg,
        marker='.', linestyle='None', color=const.colors_antAngs[ang])  # right second segment

    # plot result of ttest that 2nd segment change in position is not zero
    result1 = stats.ttest_1samp(np.squeeze(allFlyTrials_0),0,0,'omit')
    pVals1 = [result1.pvalue]
    result3 = stats.ttest_ind(np.squeeze(allFlyTrials_0), np.squeeze(allFlyTrials_2),nan_policy='omit')
    pVals3 = [result3.pvalue]

    meanIpsi = round_to_significant_digit(np.nanmean(allFlyTrials_0),2)
    meanContra = round_to_significant_digit(np.nanmean(allFlyTrials_2),2)
    axAng_single[2].text(0.95, 10, 'ipsi('+str(meanIpsi)+') ~ 0: ' +str(round_to_significant_digit(result1.pvalue,2)), color='gray')
    axAng_single[2].text(0.95, 9, 'ipsi('+str(meanIpsi)+') ~ contra('+str(meanContra)+'):\n ' +str(round_to_significant_digit(result3.pvalue,2)), color='gray')

    # plot a representation of the p-value (ns or *, **, ***)
    yMax = 7
    sft = 0
    for ii in range(np.shape(pVals3)[0]):
        yy = yMax-ii*sft
        axAng_single[2].plot([1, ii+2],[yy,yy], color=const.axisColor, linewidth=1)
        if pVals3[ii] < 0.001:
            mkr = '***'
        elif pVals3[ii] < 0.01:
            mkr = '**'
        elif pVals3[ii] < 0.05:
            mkr = '*'
        else: mkr = 'ns'
        axAng_single[2].text(ii+1.5,yy+sft/10,mkr,color=const.axisColor,fontsize=const.fontSize_axis+1)

    # configure the axes
    axAng_single[0].set_facecolor(const.figColor)
    axAng_single[0].axis('off')

    # configure the axes
    axAng_single[2].set_facecolor(const.figColor)
    set_axis_standard_preferences(axAng_single[2])
    axAng_single[2].tick_params(axis='y',colors=const.axisColor)
    axAng_single[2].spines['left'].set_color(const.axisColor)
    axAng_single[2].spines['left'].set_bounds(-10, 10)
    axAng_single[2].set_ylim([-12, 10.5])
    axAng_single[2].set_xlim([0.5, 2.5])

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig:  # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        # save histogram of angles (single trials, cross-fly)
        savepath_png = figPath + 'adaptation_distributionAngles_' + expt + '_' + cameraView + '.png'
        savepath_pdf = figPath + 'adaptation_distributionAngles_' + expt + '_' + cameraView + '.pdf'
        fig_hist.savefig(savepath_png, facecolor=fig_hist.get_facecolor())
        fig_hist.savefig(savepath_pdf, facecolor=fig_hist.get_facecolor())
        # save raw angles
        savepath_png = figPath + 'adaptation_traces_' + expt + '_' + cameraView + '.png'
        savepath_pdf = figPath + 'adaptation_traces_' + expt + '_' + cameraView + '.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())


# Get all raw traces for an entire antenna angle for particular speeds, directions
# inactivate = 1 for light on data, inactivate = 0 for light off data
# if direction=-1, will return traces from all directions (otherwise specify direction=0,1,2,3,4 for -90,-45,-,+45,+90)
def get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=1,ANGS=[1,3],speed=1,direction=-1):
    import loadNotes_mnOptoVelocityTuning as ln
    allExpts = eval('ln.notes_'+expt)

    # for each fly, grab its raw traces
    allTraces = []
    for flyNum, notes in enumerate(allExpts):
        flyExpt = notes['date']
        print('Analyizing ' + flyExpt)
        exptNotes = get_mat_notes(flyExpt)
        activate = get_activation_trials(flyExpt) # indicates if trial is light on or off

        #get info about speed, direction for all trials:
        windVel = exptNotes.velocity[:]
        windVel = windVel.str.get(0)  # take values out of brackets
        valveStates = exptNotes.valveState[:]
        windDir = valveStates.str.get(0)#/2-1  # convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
        windDir = windDir.astype(int)
        windDirs = np.unique(windDir)
        speeds = np.unique(windVel)
        uniqueValveStates = np.unique(valveStates.str.get(0))

        # plot all the single-trial angle traces for this fly
        if inactivate == 0: #light off
            if direction >= 0:
                inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (exptNotes.valveState == uniqueValveStates[direction]) & (activate == 0) == True].tolist()
            else:
                inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (activate == 0) == True].tolist()
        else: #light on
            if direction >= 0:
                inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (exptNotes.valveState == uniqueValveStates[direction]) & (activate == 1) == True].tolist()
            else:
                inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (activate == 1) == True].tolist()
        print('---------------------------------------------------------------')
        print(flyExpt)
        print(inds)
        print('---------------------------------------------------------------')
        [angs_all, _, _, _, _, _, _] = getAntennaTracesAvgs_singleExpt(flyExpt, cameraView, 0, allowFlight)
        angs_all = detect_tracking_errors_raw_trace(exptNotes, flyExpt, angs_all)
        # gather raw traces
        if flyNum == 0:
            if np.shape(ANGS)[0] > 1:
                allTraces = angs_all[inds,:,ANGS[0]]
                allTraces = np.r_[allTraces, angs_all[inds,:,ANGS[1]]]
            else:
                allTraces = angs_all[inds,:,ANGS[0]]
        else:
            if np.shape(ANGS)[0] > 1:
                allTraces = np.r_[allTraces, angs_all[inds,:,ANGS[0]]]
                allTraces = np.r_[allTraces, angs_all[inds,:,ANGS[1]]]
            else:
                allTraces = np.r_[allTraces, angs_all[inds,:,ANGS[0]]]
    return allTraces


#
def get_all_raw_traces_single_fly(flyExpt='2020_12_14_E3',cameraView='frontal',allowFlight = 1,inactivate=0,ANG=0,speed=0,direction=-1):
    import loadNotes_mnOptoVelocityTuning as ln
    # for each fly, grab its raw traces
    allTraces = []
    print('Analyizing ' + flyExpt)
    exptNotes = get_mat_notes(flyExpt)
    activate = get_activation_trials(flyExpt) # indicates if trial is light on or off

    #get info about speed, direction for all trials:
    windVel = exptNotes.velocity[:]
    windVel = windVel.str.get(0)  # take values out of brackets
    valveStates = exptNotes.valveState[:]
    windDir = valveStates.str.get(0)# convert wind direction encoding 2,4,6,8,10 to 0,1,2,3,4
    windDir = windDir.astype(int)
    windDirs = np.unique(windDir)
    speeds = np.unique(windVel)
    uniqueValveStates = np.unique(valveStates.str.get(0))
    # plot all the single-trial angle traces for this fly
    if inactivate == 0: #light off
        if direction >= 0:
            inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (exptNotes.valveState == uniqueValveStates[direction]) & (activate == 0) == True].tolist()
        else:
            inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (activate == 0) == True].tolist()
    else: #light on
        if direction >= 0:
            inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (exptNotes.valveState == uniqueValveStates[direction]) & (activate == 1) == True].tolist()
        else:
            inds = exptNotes.index[(exptNotes.velocity == speeds[speed]) & (activate == 1) == True].tolist()
    print('---------------------------------------------------------------')
    print(flyExpt)
    print(inds)
    print('---------------------------------------------------------------')
    [angs_all, _, _, _, _, _, _] = getAntennaTracesAvgs_singleExpt(flyExpt, cameraView, 0, allowFlight)
    angs_all = detect_tracking_errors_raw_trace(exptNotes, flyExpt, angs_all)
    # gather raw traces
    allTraces = angs_all[inds,:,ANG]
    return allTraces

# This is moderately inefficient but accurate way to grab and compare all of the raw traces across flies
def get_xcorr_avg_all_flies(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=1,ANG1=1,ANG2=3,speed=0,direction=-1,maxlag=284):
    import loadNotes_mnOptoVelocityTuning as ln
    allExpts = eval('ln.notes_'+expt)
    IN_LIGHT_INDS = np.arange(75,360,1)
    # for each fly, grab its raw traces
    allTraces = []
    for flyNum, notes in enumerate(allExpts):
        expt = notes['date']
        traces1 = get_all_raw_traces_single_fly(flyExpt=expt,cameraView='frontal',allowFlight = 1,inactivate=0,ANG=ANG1,speed=1,direction=-1)
        traces2 = get_all_raw_traces_single_fly(flyExpt=expt,cameraView='frontal',allowFlight = 1,inactivate=0,ANG=ANG2,speed=1,direction=-1)
        print(np.shape(traces1))

        pltCorr = get_xcorr(traces1[:,IN_LIGHT_INDS], traces2[:,IN_LIGHT_INDS], maxlag)
        print(np.shape(pltCorr))
        if flyNum == 0:
            allCorrAvg = np.nanmean(pltCorr,1)
        else: #collect array of all the autocorrs
            allCorrAvg = np.c_[allCorrAvg,  np.nanmean(pltCorr,1)]

    print(np.shape(allCorrAvg))
    return allCorrAvg

# save autocorrelation data for a given set of raw traces
# This by default takes the time during wind stimulus region (hard-coded)
# optional: save with 'traceName'
def get_autocorr(allTraces):
    TEST = 0
    stopWind = const.autocorr_stopWind
    startWind = const.autocorr_startWind
    maxlag = const.autocorr_maxlag
    if TEST:
        fig, ax = plt.subplots(1,2,facecolor=const.figColor, figsize=(8, 8))

    for ii in range(allTraces.shape[0]):
        trace = allTraces[ii,:]  #get a single raw antenna trace
        trace_stim = trace[startWind:stopWind]  # grab the region during wind
        trace_stim = trace_stim-np.nanmean(trace_stim)
        if TEST: ax[0].plot(trace_stim, color=const.med_gray)
        pltCorr = plt.acorr(trace_stim,normed=False,usevlines=False,maxlags=maxlag) #xx[1] contains correlation vector
        if ii == 0:
            allCorr = pltCorr[1]
        else: #collect array of all the autocorrs
            allCorr = np.c_[allCorr, pltCorr[1]]
    return allCorr

# returns cross-correlation across two matrices of traces
def get_xcorr(traces1, traces2, maxlag):
    for ii in range(traces1.shape[0]):
        t1 = traces1[ii,:]-np.nanmean(traces1[ii,:])  #get a single raw antenna trace
        t2 = traces2[ii,:]-np.nanmean(traces2[ii,:])  #get a single raw antenna trace
        pltCorr = plt.xcorr(t1,t2,normed=True,usevlines=False,maxlags=maxlag) #xx[1] contains correlation vector
        if ii == 0:
            allCorr = pltCorr[1]
        else: #collect array of all the autocorrs
            allCorr = np.c_[allCorr, pltCorr[1]]
    return allCorr

# test autocorr function
# will save traces, e.g. 18D07_inactivate_frontal_lightoff_all_raw_traces.npy in the SavedAngles folder
# specify experiment and direction (-1 for all directions, or 0,1,2,3,4 for -90,-45,0,+45,+90)
def test_autocorr(import_anew = 0, expt = '18D07_inactivate',direction=-1):
    SPEED = 1 #200cm/s for inactivation experiments
    ANGS = [0,2] #(left 2nd segment)
    cameraView = 'frontal'
    stopWind = const.autocorr_stopWind
    startWind = const.autocorr_startWind
    allowFlight = 1
    saved_angle_lighton_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lighton_all_raw_traces.npy'
    saved_angle_lightoff_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lightoff_all_raw_traces.npy'
    if os.path.isfile(saved_angle_lighton_fn) & (import_anew == 0):
        allTraces_lighton_18D07 = np.load(saved_angle_lighton_fn)
        allTraces_lightoff_18D07 = np.load(saved_angle_lightoff_fn)
    else:
        allTraces_lighton_18D07 = get_all_raw_traces(expt,cameraView,allowFlight,1,ANGS,SPEED)#,DIR)
        np.save(saved_angle_lighton_fn, allTraces_lighton_18D07)
        allTraces_lightoff_18D07 = get_all_raw_traces(expt,cameraView,allowFlight,0,ANGS,SPEED)#,DIR)
        np.save(saved_angle_lightoff_fn, allTraces_lightoff_18D07)
    # load/generate autocorr data
    allCorr_lighton_18D07 = get_autocorr(allTraces_lighton_18D07)
    allCorr_lightoff_18D07 = get_autocorr(allTraces_lightoff_18D07)

    fig, ax = plt.subplots(1,4,facecolor=const.figColor, figsize=(16, 10))
    fig_corr, ax_corr = plt.subplots(1,3,facecolor=const.figColor, figsize=(8, 8))
    #plot raw traces and autocorrelation for 18D07 inactivation
    print(allTraces_lighton_18D07.shape)
    for ii in range(allTraces_lighton_18D07.shape[0]):
        trace = allTraces_lightoff_18D07[ii,:]
        trace = trace-np.nanmean(trace)
        ax[0].plot(trace,color=const.med_gray,alpha=const.transparencyAntTrace)
        trace_stim = trace[startWind:stopWind]
        ax[1].plot(trace_stim,color=const.med_gray,alpha=const.transparencyAntTrace)

        trace = allTraces_lighton_18D07[ii,:]
        trace = trace-np.nanmean(trace)
        ax[2].plot(trace,color=const.med_gray,alpha=const.transparencyAntTrace)
        trace_stim = trace[startWind:stopWind]
        ax[3].plot(trace_stim,color=const.med_gray,alpha=const.transparencyAntTrace)
        ax_corr[0].plot(allCorr_lightoff_18D07[:,ii],color=const.med_gray,alpha=const.transparencyAntTrace)
        ax_corr[1].plot(allCorr_lighton_18D07[:,ii],color=const.med_gray,alpha=const.transparencyAntTrace)
    ax_corr[0].plot(np.nanmean(allCorr_lightoff_18D07,axis=1),color='black',linewidth = 3)
    ax_corr[1].plot(np.nanmean(allCorr_lighton_18D07,axis=1),color='red',linewidth = 3)

    ax_corr[2].plot(np.nanmean(allCorr_lightoff_18D07,axis=1),color='black',linewidth = 3)
    ax_corr[2].plot(np.nanmean(allCorr_lighton_18D07,axis=1),color='red',linewidth = 3)

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

# Plot autocorrelation and variance (inactivation experiments)
# import previously saved angles, or generate them anew
def plot_autocorr_and_var(cameraView='frontal',import_anew = 0,savefig=0):
    maxlag = const.autocorr_maxlag
    stopWind = const.autocorr_stopWind
    startWind = const.autocorr_startWind
    allowFlight = 1
    stimInds = list(range(startWind, stopWind))
    numPreInds = int(const.activateStart*const.framerate)
    baselineInds = list(range(0, numPreInds))
    # define which speed, direction, and antenna segment to focus on (i.e. strong deflections of one antenna)
    SPEED = 1 #200cm/s for inactivation experiments
    ANG = [0,2] #(2nd segments)
    import loadNotes_mnOptoVelocityTuning as ln

    # plot distribution of antenna angles for each antennal segment separately
    # this is not efficient given the loop above but simpler to deal with!
    fig, ax = plt.subplots(3,4,facecolor=const.figColor, figsize=(16, 10)) #plot raw traces
    fig_corr, ax_corr = plt.subplots(2,3, figsize=(10, 8)) #autocorr plots
    fig_corr.tight_layout()
    fig_var, ax_var = plt.subplots(1,4, figsize=(10, 10)) #autocorr plots
    fig_var.tight_layout()

    #load control (canton-S) raw traces
    cameraView='frontal'
    expt = 'emptyGAL4_inactivate'
    saved_angle_lighton_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lighton_all_raw_traces.npy'
    saved_corr_lighton_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lighton_autocorr.npy'
    saved_angle_lightoff_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lightoff_all_raw_traces.npy'
    saved_corr_lightoff_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lightoff_autocorr.npy'
    if os.path.isfile(saved_angle_lighton_fn) & (import_anew == 0):
        allTraces_lighton_ctrl = np.load(saved_angle_lighton_fn)
        allTraces_lightoff_ctrl = np.load(saved_angle_lightoff_fn)
    else:
        allTraces_lighton_ctrl = get_all_raw_traces(expt,cameraView,allowFlight,1,ANG,SPEED)#,DIR)
        np.save(saved_angle_lighton_fn, allTraces_lighton_ctrl)
        allTraces_lightoff_ctrl = get_all_raw_traces(expt,cameraView,allowFlight,0,ANG,SPEED)#,DIR)
        np.save(saved_angle_lightoff_fn, allTraces_lightoff_ctrl)
    # load/generate autocorr data
    if os.path.isfile(saved_corr_lighton_fn) & (import_anew == 0):
        allCorr_lighton_ctrl = np.load(saved_corr_lighton_fn)
        allCorr_lightoff_ctrl = np.load(saved_corr_lightoff_fn)
    else:
        allCorr_lighton_ctrl = get_autocorr(allTraces_lighton_ctrl)
        np.save(saved_corr_lighton_fn, allCorr_lighton_ctrl)
        allCorr_lightoff_ctrl = get_autocorr(allTraces_lightoff_ctrl)
        np.save(saved_corr_lightoff_fn, allCorr_lightoff_ctrl)

    #load 18D07 raw traces
    expt = '18D07_inactivate'
    saved_angle_lighton_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lighton_all_raw_traces.npy'
    saved_corr_lighton_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lighton_autocorr.npy'
    saved_angle_lightoff_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lightoff_all_raw_traces.npy'
    saved_corr_lightoff_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lightoff_autocorr.npy'
    if os.path.isfile(saved_angle_lighton_fn) & (import_anew == 0):
        allTraces_lighton_18D07 = np.load(saved_angle_lighton_fn)
        allTraces_lightoff_18D07 = np.load(saved_angle_lightoff_fn)
    else:
        allTraces_lighton_18D07 = get_all_raw_traces(expt,cameraView,allowFlight,1,ANG,SPEED)#,DIR)
        np.save(saved_angle_lighton_fn, allTraces_lighton_18D07)
        allTraces_lightoff_18D07 = get_all_raw_traces(expt,cameraView,allowFlight,0,ANG,SPEED)#,DIR)
        np.save(saved_angle_lightoff_fn, allTraces_lightoff_18D07)
    # load/generate autocorr data
    if os.path.isfile(saved_corr_lighton_fn) & (import_anew == 0):
        allCorr_lighton_18D07 = np.load(saved_corr_lighton_fn)
        allCorr_lightoff_18D07 = np.load(saved_corr_lightoff_fn)
    else:
        allCorr_lighton_18D07 = get_autocorr(allTraces_lighton_18D07)
        np.save(saved_corr_lighton_fn, allCorr_lighton_18D07)
        allCorr_lightoff_18D07 = get_autocorr(allTraces_lightoff_18D07)
        np.save(saved_corr_lightoff_fn, allCorr_lightoff_18D07)

    #load 91F02 raw traces
    expt = '91F02_inactivate'
    saved_angle_lighton_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lighton_all_raw_traces.npy'
    saved_corr_lighton_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lighton_autocorr.npy'
    saved_angle_lightoff_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lightoff_all_raw_traces.npy'
    saved_corr_lightoff_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'lightoff_autocorr.npy'
    if os.path.isfile(saved_angle_lighton_fn) & (import_anew == 0):
        allTraces_lighton_91F02 = np.load(saved_angle_lighton_fn)
        allTraces_lightoff_91F02 = np.load(saved_angle_lightoff_fn)
    else:
        allTraces_lighton_91F02 = get_all_raw_traces(expt,cameraView,allowFlight,1,ANG,SPEED)#,DIR)
        np.save(saved_angle_lighton_fn, allTraces_lighton_91F02)
        allTraces_lightoff_91F02 = get_all_raw_traces(expt,cameraView,allowFlight,0,ANG,SPEED)#,DIR)
        np.save(saved_angle_lightoff_fn, allTraces_lightoff_91F02)
    # load/generate autocorr data
    if os.path.isfile(saved_corr_lighton_fn) & (import_anew == 0):
        allCorr_lighton_91F02 = np.load(saved_corr_lighton_fn)
        allCorr_lightoff_91F02 = np.load(saved_corr_lightoff_fn)
    else:
        allCorr_lighton_91F02 = get_autocorr(allTraces_lighton_91F02)
        np.save(saved_corr_lighton_fn, allCorr_lighton_91F02)
        allCorr_lightoff_91F02 = get_autocorr(allTraces_lightoff_91F02)
        np.save(saved_corr_lightoff_fn, allCorr_lightoff_91F02)

    #plot raw traces and autocorrelation for Canton-S controls
    for ii in range(allTraces_lightoff_ctrl.shape[0]-2):
        #plot light off data
        trace = allTraces_lightoff_ctrl[ii,:]
        trace = trace-np.nanmean(trace)
        ax[0,0].plot(trace,color=const.med_gray,alpha=const.transparencyAntTrace)
        trace_stim = trace[startWind:stopWind]
        ax[0,1].plot(trace_stim,color=const.med_gray,alpha=const.transparencyAntTrace)
    for ii in range(allTraces_lighton_ctrl.shape[0]-2):
        # plot light on data
        trace = allTraces_lighton_ctrl[ii,:]
        trace = trace-np.nanmean(trace)
        ax[0,2].plot(trace,color=const.med_gray,alpha=const.transparencyAntTrace)
        trace_stim = trace[startWind:stopWind]
        ax[0,3].plot(trace_stim,color=const.med_gray,alpha=const.transparencyAntTrace)

    #reconfigure this plot to show variance within a single expt (e.g. 18D07 inactivation light vs. no light)
    varWholeTrace = np.nanvar(allTraces_lightoff_ctrl,axis=0)
    baseline = np.nanmean(varWholeTrace[baselineInds])
    var_lightoff_ctrl = varWholeTrace-baseline
    ax_var[0].plot(var_lightoff_ctrl,color='black',linewidth = 1,alpha=const.transparencyCrossFlyTrace) #plot variance during wind (compared to pre-light baseline)
    ax_var[1].plot(var_lightoff_ctrl[startWind:stopWind],color='black',linewidth = 1,alpha=const.transparencyCrossFlyTrace) #plot variance during wind (compared to pre-light baseline)
    varWholeTrace = np.nanvar(allTraces_lighton_ctrl,axis=0)
    baseline = np.nanmean(varWholeTrace[baselineInds])
    var_lighton_ctrl = varWholeTrace-baseline
    ax_var[2].plot(var_lighton_ctrl,color='black',linewidth = 1,alpha=const.transparencyCrossFlyTrace) #plot variance during wind (compared to pre-light baseline)
    ax_var[3].plot(var_lighton_ctrl[startWind:stopWind],color='black',linewidth = 1,alpha=const.transparencyCrossFlyTrace) #plot variance during wind (compared to pre-light baseline)

    #plot raw traces and autocorrelation for 91F02 inactivation
    for ii in range(allTraces_lightoff_91F02.shape[0]-1):
        trace = allTraces_lightoff_91F02[ii,:]
        trace = trace-np.nanmean(trace)
        ax[1,0].plot(trace,color=const.med_gray,alpha=const.transparencyAntTrace)
        trace_stim = trace[startWind:stopWind]
        ax[1,1].plot(trace_stim,color=const.med_gray,alpha=const.transparencyAntTrace)
    for ii in range(allTraces_lighton_91F02.shape[0]-1):
        trace = allTraces_lighton_91F02[ii,:]
        trace = trace-np.nanmean(trace)
        ax[1,2].plot(trace,color=const.med_gray,alpha=const.transparencyAntTrace)
        trace_stim = trace[startWind:stopWind]
        ax[1,3].plot(trace_stim,color=const.med_gray,alpha=const.transparencyAntTrace)

        ax_corr[0,0].plot(allCorr_lighton_91F02[:,ii],color=const.med_gray,alpha=const.transparencyAntTrace)
        ax_corr[0,1].plot(allCorr_lighton_91F02[:,ii],color=const.med_gray,alpha=const.transparencyAntTrace)

    ax_corr[0,0].plot(np.nanmean(allCorr_lightoff_91F02,axis=1),color='black',linewidth = 3)
    ax_corr[0,1].plot(np.nanmean(allCorr_lighton_91F02,axis=1),color='red',linewidth = 3)

    varWholeTrace = np.nanvar(allTraces_lightoff_91F02,axis=0)
    baseline = np.nanmean(varWholeTrace[baselineInds])
    var_91F02 = varWholeTrace-baseline
    ax_var[0].plot(varWholeTrace-baseline,color='blue',linewidth = 1,alpha=const.transparencyCrossFlyTrace)  #plot variance during wind (compared to pre-light baseline)
    ax_var[1].plot(var_91F02[startWind:stopWind],color='blue',linewidth = 1,alpha=const.transparencyCrossFlyTrace) #plot variance during wind (compared to pre-light baseline)

    varWholeTrace = np.nanvar(allTraces_lighton_91F02,axis=0)
    baseline = np.nanmean(varWholeTrace[baselineInds])
    var_91F02 = varWholeTrace-baseline
    ax_var[2].plot(varWholeTrace-baseline,color='blue',linewidth = 1,alpha=const.transparencyCrossFlyTrace)  #plot variance during wind (compared to pre-light baseline)
    ax_var[3].plot(var_91F02[startWind:stopWind],color='blue',linewidth = 1,alpha=const.transparencyCrossFlyTrace) #plot variance during wind (compared to pre-light baseline)


    #plot raw traces and autocorrelation for 18D07 inactivation
    for ii in range(allTraces_lighton_18D07.shape[0]):
        trace = allTraces_lightoff_18D07[ii,:]
        trace = trace-np.nanmean(trace)
        ax[2,0].plot(trace,color=const.med_gray,alpha=const.transparencyAntTrace)
        trace_stim = trace[startWind:stopWind]
        ax[2,1].plot(trace_stim,color=const.med_gray,alpha=const.transparencyAntTrace)

        trace = allTraces_lighton_18D07[ii,:]
        trace = trace-np.nanmean(trace)
        ax[2,2].plot(trace,color=const.med_gray,alpha=const.transparencyAntTrace)
        trace_stim = trace[startWind:stopWind]
        ax[2,3].plot(trace_stim,color=const.med_gray,alpha=const.transparencyAntTrace)

        ax_corr[1,0].plot(allCorr_lightoff_18D07[:,ii],color=const.med_gray,alpha=const.transparencyAntTrace)
        ax_corr[1,1].plot(allCorr_lighton_18D07[:,ii],color=const.med_gray,alpha=const.transparencyAntTrace)
    ax_corr[1,0].plot(np.nanmean(allCorr_lightoff_18D07,axis=1),color='black',linewidth = 3)
    ax_corr[1,1].plot(np.nanmean(allCorr_lighton_18D07,axis=1),color='red',linewidth = 3)

    varWholeTrace = np.nanvar(allTraces_lightoff_18D07,axis=0)
    baseline = np.nanmean(varWholeTrace[baselineInds])
    var_18D07 = varWholeTrace-baseline
    ax_var[0].plot(varWholeTrace-baseline,color='green',linewidth = 1,alpha=const.transparencyCrossFlyTrace)  #plot variance during wind (compared to pre-light baseline)
    ax_var[1].plot(var_18D07[startWind:stopWind],color='green',linewidth = 1,alpha=const.transparencyCrossFlyTrace) #plot variance during wind (compared to pre-light baseline)
    varWholeTrace = np.nanvar(allTraces_lighton_18D07,axis=0)
    baseline = np.nanmean(varWholeTrace[baselineInds])
    var_18D07 = varWholeTrace-baseline
    ax_var[2].plot(varWholeTrace-baseline,color='green',linewidth = 1,alpha=const.transparencyCrossFlyTrace)  #plot variance during wind (compared to pre-light baseline)
    ax_var[3].plot(var_18D07[startWind:stopWind],color='green',linewidth = 1,alpha=const.transparencyCrossFlyTrace) #plot variance during wind (compared to pre-light baseline)

    # plot average autocorrelation on top of one another
    ax_corr[0,2].plot(np.nanmean(allCorr_lightoff_91F02,axis=1),color='black',linewidth = 3)
    ax_corr[0,2].plot(np.nanmean(allCorr_lighton_91F02,axis=1),color='red',linewidth = 3)

    ax_corr[1,2].plot(np.nanmean(allCorr_lightoff_18D07,axis=1),color='black',linewidth = 3)
    ax_corr[1,2].plot(np.nanmean(allCorr_lighton_18D07,axis=1),color='red',linewidth = 3)

    ax_corr[0,2].text(38,0.8,'light off', color='black')
    ax_corr[0,2].text(38,0.88,'light on', color='red')

    # configure axes for autocorr average comparison
    #set_acorr_axes(ax_corr[0,0],maxlag,[0.45, 1.01],[0.5,1],['-0.5','0','0.5'])
    #set_acorr_axes(ax_corr[0,1],maxlag,[0.45, 1.01],[0.5,1],['-0.5','0','0.5'])
    #set_acorr_axes(ax_corr[0,2],maxlag,[0.45, 1.01],[0.5,1],['-0.5','0','0.5'])
    set_acorr_axes(ax_corr[0,0],maxlag,[0.15, 1.01],[0.2,1],['-0.5','0','0.5'])
    set_acorr_axes(ax_corr[0,1],maxlag,[0.15, 1.01],[0.2,1],['-0.5','0','0.5'])
    set_acorr_axes(ax_corr[0,2],maxlag,[0.15, 1.01],[0.2,1],['-0.5','0','0.5'])
    set_acorr_axes(ax_corr[1,0],maxlag,[0.15, 1.01],[0.2,1],['-0.5','0','0.5'])
    set_acorr_axes(ax_corr[1,1],maxlag,[0.15, 1.01],[0.2,1],['-0.5','0','0.5'])
    set_acorr_axes(ax_corr[1,2],maxlag,[0.15, 1.01],[0.2,1],['-0.5','0','0.5'])

    #ax_corr[0,0].set_title('control (emptyGAL4)', color='black')
    ax_corr[0,0].set_title('light off', color='black')
    ax_corr[0,1].set_title('light on', color='black')
    ax_corr[0,2].set_title('light off vs. on', color='black')
    ax_corr[0,0].set_ylabel('91F02_GAL4')
    ax_corr[1,0].set_ylabel('18D07_GAL4')

    ax_var[0].set_ylim([-5,100])
    ax_var[1].set_ylim([-5,100])
    ax_var[2].set_ylim([-5,100])
    ax_var[3].set_ylim([-5,100])
    ax_var[1].text(50,50,'control (emptyGAL4)', color='black')
    ax_var[1].text(50,48,'91F02 silenced', color='blue')
    ax_var[1].text(50,46,'18D07 silenced', color='green')

    ax[0,0].set_ylabel('control (emptyGAL4)')
    ax[1,0].set_ylabel('91F02 inactivate')
    ax[2,0].set_ylabel('18D07 inactivate')
    ax[0,0].set_title('full raw traces (light OFF)')
    ax[0,1].set_title('during wind (light OFF)')
    ax[0,2].set_title('full raw traces (light ON)')
    ax[0,3].set_title('during wind (light ON)')

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig:

        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        # save histogram of angles (single trials, cross-fly)
        savepath_png = figPath + 'rawTracesForCorr_' + '_' + cameraView + '.png'
        savepath_pdf = figPath + 'rawTracesForCorr_' + '_' + cameraView + '.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())

        # save histogram of angles (single trials, cross-fly)
        savepath_png = figPath + 'autocorr_' + '_' + cameraView + '.png'
        savepath_pdf = figPath + 'autocorr_' + '_' + cameraView + '.pdf'
        fig_corr.savefig(savepath_png, facecolor=fig_corr.get_facecolor())
        fig_corr.savefig(savepath_pdf, facecolor=fig_corr.get_facecolor())
        # save raw angles
        savepath_png = figPath + 'variance_' + cameraView + '.png'
        savepath_pdf = figPath + 'variance_' + cameraView + '.pdf'
        fig_var.savefig(savepath_png, facecolor=fig_var.get_facecolor())
        fig_var.savefig(savepath_pdf, facecolor=fig_var.get_facecolor())


# Plot cross-correlation of left and right antennnae
def plot_xcorr_no_wind(cameraView='frontal',import_anew = 0,savefig=0):
    LESS_PLOT = 1 #if 1, will plot only data used in manuscript figure (it's so huge with all the traces not downsampled...)

    maxlag = 284#const.autocorr_maxlag
    IN_LIGHT_INDS = np.arange(75,360,1) #compare traces within light region (exclude inds at ends, where coordinated response to light on/off may occur)
    import loadNotes_mnOptoVelocityTuning as ln
    #NO_WIND_INDS = np.concatenate([np.arange(0,59,1),np.arange(65,119,1),np.arange(245,479,1)])
    allowFlight = 1
    # define which speed, direction, and antenna segment to focus on
    SPEED = 0 #0cm/s (activation experiment has 0, 50, 100, 200 cm/s)
    #angle indices:
    R2 = 0
    R3 = 1
    L2 = 2
    L3 = 3

    #load control (canton-S) raw traces
    cameraView='frontal'
    expt = 'CS_activate'
    saved_angle_CS_thirdSegR_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_no_wind_traces_thirdSegR.npy'
    saved_angle_CS_thirdSegL_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_no_wind_traces_thirdSegL.npy'
    saved_corr_CS_thirdSeg_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_autocorr_thirdSeg.npy'

    saved_angle_CS_secondSegR_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_no_wind_traces_secondSegR.npy'
    saved_angle_CS_secondSegL_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_no_wind_traces_secondSegL.npy'
    saved_corr_CS_secondSeg_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_autocorr_secondSeg.npy'

    saved_randCorr_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_autocorr_thirdRRand.npy'
    saved_thirdThirdCorr_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_autocorr_thirdThird.npy'
    saved_secondSecondCorr_fn = const.savedDataDirectory+expt+'_'+cameraView+'_'+'CS_activate_autocorr_secondSecond.npy'

    if os.path.isfile(saved_angle_CS_thirdSegR_fn) & (import_anew == 0):
        allTraces_CS_thirdSegR = np.load(saved_angle_CS_thirdSegR_fn)
        allTraces_CS_thirdSegL = np.load(saved_angle_CS_thirdSegL_fn)

        allTraces_CS_secondSegR = np.load(saved_angle_CS_secondSegR_fn)
        allTraces_CS_secondSegL = np.load(saved_angle_CS_secondSegL_fn)
    else:
        allTraces_CS_thirdSegR = get_all_raw_traces(expt,cameraView,allowFlight,0,[R3],SPEED)
        allTraces_CS_thirdSegR = allTraces_CS_thirdSegR[:,IN_LIGHT_INDS]
        np.save(saved_angle_CS_thirdSegR_fn, allTraces_CS_thirdSegR)

        allTraces_CS_thirdSegL = get_all_raw_traces(expt,cameraView,allowFlight,0,[L3],SPEED)
        allTraces_CS_thirdSegL = allTraces_CS_thirdSegL[:,IN_LIGHT_INDS]
        np.save(saved_angle_CS_thirdSegL_fn, allTraces_CS_thirdSegL)

        allTraces_CS_secondSegR = get_all_raw_traces(expt,cameraView,allowFlight,0,[R2],SPEED)
        allTraces_CS_secondSegR = allTraces_CS_secondSegR[:,IN_LIGHT_INDS]
        np.save(saved_angle_CS_secondSegR_fn, allTraces_CS_secondSegR)

        allTraces_CS_secondSegL = get_all_raw_traces(expt,cameraView,allowFlight,0,[L2],SPEED)
        allTraces_CS_secondSegL = allTraces_CS_secondSegL[:,IN_LIGHT_INDS]
        np.save(saved_angle_CS_secondSegL_fn, allTraces_CS_secondSegL)

    # load/generate corr data
    if os.path.isfile(saved_corr_CS_thirdSeg_fn) & (import_anew == 0):

        allCorr_CS_thirdSeg = np.load(saved_corr_CS_thirdSeg_fn)
        allCorr_CS_secondSeg = np.load(saved_corr_CS_secondSeg_fn)
        randCorr = np.load(saved_randCorr_fn)
        secondSecondCorr = np.load(saved_secondSecondCorr_fn)
        thirdThirdCorr = np.load(saved_thirdThirdCorr_fn)
    else:
        left = allTraces_CS_thirdSegL
        right = allTraces_CS_thirdSegR
        allCorr_CS_thirdSeg = get_xcorr(left, right, maxlag)
        np.save(saved_corr_CS_thirdSeg_fn, allCorr_CS_thirdSeg)

        rand = right[:, np.random.permutation(right.shape[1])] #randomly shuffle right angles
        randCorr = get_xcorr(rand, right, maxlag)
        np.save(saved_randCorr_fn, randCorr)

        secondSecondCorr = get_xcorr(allTraces_CS_secondSegL, allTraces_CS_secondSegL, maxlag)
        np.save(saved_secondSecondCorr_fn, secondSecondCorr)
        thirdThirdCorr = get_xcorr(allTraces_CS_thirdSegL, allTraces_CS_thirdSegL, maxlag)
        np.save(saved_thirdThirdCorr_fn, thirdThirdCorr)

        left = allTraces_CS_secondSegL
        right = allTraces_CS_secondSegR
        allCorr_CS_secondSeg = get_xcorr(left, right, maxlag)
        np.save(saved_corr_CS_secondSeg_fn, allCorr_CS_secondSeg)

    fig, ax = plt.subplots(2,3, figsize=(12, 18)) #autocorr plots
    fig.tight_layout()

    #left versus right second segment
    ax[0,0].plot(allCorr_CS_secondSeg,color=const.med_gray,alpha=const.transparencyAllSegmentTraces)
    ax[0,0].plot(np.nanmean(allCorr_CS_secondSeg,1), color='black')
    avgCorr_allFlies_L2_R2 = get_xcorr_avg_all_flies(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANG1=0,ANG2=2,speed=SPEED,direction=-1,maxlag=maxlag)
    ax[0,0].plot(avgCorr_allFlies_L2_R2)

    #left second versus left third segment
    print("SOMETHING UPDATING")

    if LESS_PLOT == 0:
        ax[0,1].plot(allCorr_CS_thirdSeg,color=const.med_gray,alpha=const.transparencyAllSegmentTraces)
        ax[0,1].plot(np.nanmean(allCorr_CS_thirdSeg,1), color='black')
    if LESS_PLOT  == 0:
        ax[0,2].plot(randCorr,color=const.med_gray,alpha=const.transparencyAntTrace)
        ax[0,2].plot(np.nanmean(randCorr,1), color='black')

    #left versus left second segment
    ax[1,0].plot(secondSecondCorr,color=const.med_gray,alpha=const.transparencyAllSegmentTraces)
    ax[1,0].plot(np.nanmean(secondSecondCorr,1), color='black')
    avgCorr_allFlies_L2_L2 = get_xcorr_avg_all_flies(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANG1=2,ANG2=2,speed=SPEED,direction=-1,maxlag=maxlag)
    ax[1,0].plot(avgCorr_allFlies_L2_L2)

    #left versus left third segment
    if LESS_PLOT == 0:
        ax[1,1].plot(thirdThirdCorr,color=const.med_gray,alpha=const.transparencyAllSegmentTraces)
        ax[1,1].plot(np.nanmean(thirdThirdCorr,1), color='black')

    # compute half-max width of average cross-correlation
    halfMax = np.nanmax(np.nanmean(allCorr_CS_secondSeg,1))/2
    halfMaxWid = np.squeeze(np.where(np.nanmean(allCorr_CS_secondSeg,1)>halfMax))
    halfMaxWid = halfMaxWid[-1]-halfMaxWid[0]
    halfMaxWid = halfMaxWid/const.framerate
    print('Half max width (s) ' + str(halfMaxWid))
    ax[0,0].text(1,0.99,'Half max width (s) ' + str(round_to_significant_digit(halfMaxWid,sig=3)))
    #half-max width of individual flies
    halfMax = np.nanmax(avgCorr_allFlies_L2_R2,0)/2
    hmw_L2_R2 = np.zeros(np.size(halfMax))
    for ii in range(np.size(halfMax)):
        overHalf = np.squeeze(np.where(avgCorr_allFlies_L2_R2[:,ii]>halfMax[ii]))
        hmw_L2_R2[ii] = (overHalf[-1]-overHalf[0])/const.framerate
    print('hmw_L2_R2:')
    print(hmw_L2_R2)

    # half-max for l/r third segment:
    halfMax = np.nanmax(np.nanmean(allCorr_CS_thirdSeg,1))/2
    halfMaxWid = np.squeeze(np.where(np.nanmean(allCorr_CS_thirdSeg,1)>halfMax))
    halfMaxWid = halfMaxWid[-1]-halfMaxWid[0]
    halfMaxWid = halfMaxWid/const.framerate
    print('Half max width (s) ' + str(halfMaxWid))
    ax[0,1].text(1,0.99,'Half max width (s) ' + str(round_to_significant_digit(halfMaxWid,sig=3)))

    # half-max
    halfMax = np.nanmax(np.nanmean(thirdThirdCorr,1))/2
    halfMaxWid = np.squeeze(np.where(np.nanmean(thirdThirdCorr,1)>halfMax))
    halfMaxWid = halfMaxWid[-1]-halfMaxWid[0]
    halfMaxWid = halfMaxWid/const.framerate
    print('Half max width (s) ' + str(halfMaxWid))
    ax[1,0].text(1,0.99,'Half max width (s) ' + str(round_to_significant_digit(halfMaxWid,sig=3)))

    # half-max
    halfMax = np.nanmax(np.nanmean(secondSecondCorr,1))/2
    halfMaxWid = np.squeeze(np.where(np.nanmean(secondSecondCorr,1)>halfMax))
    halfMaxWid = halfMaxWid[-1]-halfMaxWid[0]
    halfMaxWid = halfMaxWid/const.framerate
    print('Half max width (s) ' + str(halfMaxWid))
    ax[1,1].text(1,0.99,'Half max width (s) ' + str(round_to_significant_digit(halfMaxWid,sig=3)))

    halfMax = np.nanmax(avgCorr_allFlies_L2_L2,0)/2
    hmw_L2_L2 = np.zeros(np.size(halfMax))
    for ii in range(np.size(halfMax)):
        overHalf = np.squeeze(np.where(avgCorr_allFlies_L2_L2[:,ii]>halfMax[ii]))
        hmw_L2_L2[ii] = (overHalf[-1]-overHalf[0])/const.framerate
    print('hmw_L2_L2:')
    print(hmw_L2_L2)

    result = stats.ttest_rel(hmw_L2_R2, hmw_L2_L2)
    ax[1,0].text(1, 0.8, 'L2_R2 vs. L2_L2 half wid: ' +str(round_to_significant_digit(result.pvalue,2)), color='black')
    ax[0,0].text(1, 0.9, 'L2_R2 std: ' +str(round_to_significant_digit(np.nanstd(hmw_L2_R2),2)), color='black')
    ax[1,0].text(1, 0.9, 'L2_L2 std: ' +str(round_to_significant_digit(np.nanstd(hmw_L2_L2),2)), color='black')

    # configure axes for autocorr average comparison
    #(ax,maxlag,ylim,ybounds,xticklabels)
    set_acorr_axes(ax[0,0],maxlag,[-0.81, 1.1],[-0.75,1],['-4.75','0','4.75'])
    set_acorr_axes(ax[0,1],maxlag,[-0.81, 1.1],[-0.75,1],['-4.75','0','4.75'])
    set_acorr_axes(ax[0,2],maxlag,[-0.81, 1.1],[-0.75,1],['-4.75','0','4.75'])
    set_acorr_axes(ax[1,0],maxlag,[-0.81, 1.1],[-0.75,1],['-4.75','0','4.75'])
    set_acorr_axes(ax[1,1],maxlag,[-0.81, 1.1],[-0.75,1],['-4.75','0','4.75'])

    ax[0,0].set_title('l vs r 2nd segment, n='+str(np.shape(allTraces_CS_thirdSegL)[0]), color='black')
    ax[0,1].set_title('l vs r 3rd segment', color='black')
    ax[0,2].set_title('rand vs r 3rd segment', color='black')

    ax[1,0].set_title('l 2nd vs l 2nd segment', color='black')
    ax[1,1].set_title('l 3rd vs l 3rd segment', color='black')
    ax[1,2].clear() #clear output in one of the subfigures from running xcorr

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig:
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        # save histogram of angles (single trials, cross-fly)
        savepath_png = figPath + 'crosscorr_' + cameraView + '.png'
        savepath_pdf = figPath + 'crosscorr_' + cameraView + '.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())
# save some raw traces in .mat format for Kathy to use!
# 3/3/22


# save tuning data .mat format for Kathy to use!
# 3/3/22
def save_npy_to_mat_wind_vel_tuning():

    cameraView = 'frontal'
    importAnew = 0 #this will extra angles from scratch (takes forever, only do if we change some base property of this)
    extraText = ''
    filesToConvert = [const.savedDataDirectory+'CS_activate'+'_'+cameraView+extraText+'RminL_allTrials.npy',
    const.savedDataDirectory+'18D07_activate'+'_'+cameraView+extraText+'RminL_allTrials.npy',
    const.savedDataDirectory+'91F02_activate'+'_'+cameraView+extraText+'RminL_allTrials.npy',
    const.savedDataDirectory+'74C10_activate'+'_'+cameraView+extraText+'RminL_allTrials.npy']

    for ind, fn in enumerate(filesToConvert):
        print(ind)
        print(fn[:-4])
        data = np.load(fn)
        mat_fn = fn[:-4]+'.mat'
        savemat(mat_fn,{'data':data})





# plot displacement of the second antennal segment during wind, across directions, for 200 cm/s
# For Canton-S (activation) control flies
def plot_second_displacement(importAnew = 0,savefig = 0):

    directions = [0,1,2,3,4]
    directionsLeft = [4,3,2,1,0]
    rangeTrace = range(70,350,1) #area around wind stimulus (all during light)
    #stimulus bar constants
    rectHeight_wind = 0.5
    ymin = -10
    rectX_wind = 50  # -1 to align with 0 indexing
    rectY_wind = ymin-rectHeight_wind*2
    rectWid_wind = int(2*const.framerate)

    # Plot the data
    fig, ax = plt.subplots(3,6,facecolor=const.figColor, figsize=(22, 18))
    fig2, ax2 = plt.subplots(2,2,facecolor=const.figColor, figsize=(8, 8))
    rightBase3 = np.empty([])
    rightDeflect3 = np.empty([])
    rightBase2 = np.empty([])
    rightDeflect2 = np.empty([])
    leftBase3, leftDeflect3, leftBase2, leftDeflect2 = [], [], [], []

    for dir in directions:
        fn_right2 = const.savedDataDirectory+'traces_CS_activate_200_right2_'+str(dir)+'.npy'
        fn_right3 = const.savedDataDirectory+'traces_CS_activate_200_right3_'+str(dir)+'.npy'
        fn_left2 = const.savedDataDirectory+'traces_CS_activate_200_left2_'+str(directionsLeft[dir])+'.npy' #flip left onto right later (assuming symmetry)
        fn_left3 = const.savedDataDirectory+'traces_CS_activate_200_left3_'+str(directionsLeft[dir])+'.npy'
        if ~importAnew & os.path.isfile(fn_right2):
            traces_right2 = np.load(fn_right2)
            traces_right3 = np.load(fn_right3)
            traces_left2 = np.load(fn_left2)
            traces_left3 = np.load(fn_left3)
        else:
            traces_right2 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[0],speed=3,direction=dir)
            np.save(fn_right2, traces_right2)
            traces_right3 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[1],speed=3,direction=dir)
            np.save(fn_right3, traces_right3)
            traces_left2 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[2],speed=3,direction=dir)
            np.save(fn_left2, traces_left2)
            traces_left3 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[3],speed=3,direction=dir)
            np.save(fn_left3, traces_left3)
            print('Importing traces:\n'+fn_right2)


        # tally up all traces, for a full cross-direction second segment estimate
        if dir == 0: #initialize cross-dir structure
            traces_2_all = traces_right2
        else: #tally up all second segment traces
            traces_2_all = np.concatenate((traces_2_all,traces_right2),axis=0)
        # plot deflections for each direction here
        for ii in range(np.shape(traces_right2)[0]):
            trace2 = traces_right2[ii,rangeTrace]
            trace2 = trace2-np.nanmean(trace2[0:45])
            ax[0,dir].plot(trace2,color=const.med_gray,alpha=const.transparencyAllSegmentTraces,linewidth=const.traceWidRaw)
            ax[0,dir].set_title(const.windDirLabels[dir]+' deg')

            trace3 = traces_right3[ii,rangeTrace]
            trace3 = trace3-np.nanmean(trace3[0:45])
            ax[1,dir].plot(trace3,color=const.med_gray,alpha=const.transparencyAllSegmentTraces,linewidth=const.traceWidRaw)
            ax[1,dir].set_title(const.windDirLabels[dir]+' deg')
            # plot 3rd vs. 2nd on right for all directions
            ax[0,5].plot(trace3,trace2,'.',color=const.med_gray, alpha=const.transparencyAllActiveTraces)
            # plot 3rd vs. 2nd for single directions below
            ax[2,dir].plot(trace3,trace2,'.',color=const.med_gray, alpha=const.transparencyAllActiveTraces)

            if dir == 0:
                trace_right2 = traces_right2[ii,rangeTrace] #grab within light region
                baseline = np.nanmean(trace_right2[0:45])
                deflection = np.nanmean(trace_right2[53:165])-baseline
                rightBase2 = np.append(rightBase2,baseline)

                rightDeflect2 = np.append(rightDeflect2,deflection)
                trace_right3 = traces_right3[ii,rangeTrace] #grab within light region
                baseline = np.nanmean(trace_right3[0:45])
                deflection = np.nanmean(trace_right3[53:165])-baseline
                rightBase3 = np.append(rightBase3,baseline)
                rightDeflect3 = np.append(rightDeflect3,deflection)

        ax[2,dir].set_ylim([-15,25])
        ax[2,dir].set_xlim([-25,20])
        ax[2,dir].set_ylabel('second seg')
        ax[2,dir].set_xlabel('third seg')

        for ii in range(np.shape(traces_left2)[0]):
            trace2 = traces_left2[ii,rangeTrace]
            trace2 = trace2-np.nanmean(trace2[0:45])
            ax[0,dir].plot(trace2,color=const.salmon,alpha=const.transparencyAllSegmentTraces,linewidth=const.traceWidRaw)
            ax[0,dir].set_title(const.windDirLabels[dir]+' deg')

            trace3 = traces_left3[ii,rangeTrace]
            trace3 = trace3-np.nanmean(trace3[0:45])
            ax[1,dir].plot(trace3,color=const.salmon,alpha=const.transparencyAllSegmentTraces,linewidth=const.traceWidRaw)
            ax[1,dir].set_title(const.windDirLabels[dir]+' deg')
            # plot 3rd vs. 2nd on right
            ax[2,dir].plot(trace3,trace2,'.',color=const.salmon, alpha=const.transparencyAllActiveTraces)

            if dir == 0:
                trace_left2 = traces_left2[ii,rangeTrace] #grab within light region
                baseline = np.nanmean(trace_left2[0:45])
                deflection = np.nanmean(trace_left2[53:165])-baseline
                leftBase2 = np.append(leftBase2,baseline)
                leftDeflect2 = np.append(leftDeflect2,deflection)
                trace_left3 = traces_left3[ii,rangeTrace] #grab within light region
                baseline = np.nanmean(trace_left3[0:45])
                deflection = np.nanmean(trace_left3[53:165])-baseline
                leftBase3 = np.append(leftBase3,baseline)
                leftDeflect3 = np.append(leftDeflect3,deflection)

        # combine flipped left and right traces into one structure
        traces_right2 = np.concatenate((traces_right2,traces_left2),axis=0)
        traces_right3 = np.concatenate((traces_right3,traces_left3),axis=0)

        #plot average deflection
        avgTrace2 = np.nanmean(traces_right2[:,rangeTrace],0)
        avgTrace2 = avgTrace2-np.nanmean(avgTrace2[0:45])
        ax[0,dir].plot(avgTrace2,color=const.dark_gray,linewidth=const.traceWidAvg)
        avgTrace3 = np.nanmean(traces_right3[:,rangeTrace],0)
        avgTrace3 = avgTrace3-np.nanmean(avgTrace3[0:45])
        ax[1,dir].plot(avgTrace3,color=const.dark_gray,linewidth=const.traceWidAvg)

        # configure the axes
        ax[0,dir].set_facecolor(const.figColor)
        ax[0,dir].axis('off')
        ax[0,dir].set_facecolor(const.figColor)
        ax[0,dir].set_ylim([-25, 25])
        ax[0,dir].set_xlim([-5, 290])
        ax[1,dir].set_facecolor(const.figColor)
        ax[1,dir].axis('off')
        ax[1,dir].set_facecolor(const.figColor)
        ax[1,dir].set_ylim([-25, 25])
        ax[1,dir].set_xlim([-5, 290])

        # draw wind stimulus bar
        ax[0,dir].add_patch(Rectangle((rectX_wind, rectY_wind), rectWid_wind, rectHeight_wind,
            facecolor=const.axisColor))
        ax[0,dir].text(rectX_wind+rectWid_wind/2, rectY_wind-rectHeight_wind*3-const.fontSize_stimBar/const.scaleAng_rawTraces,
            str(2)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='center',
            color=const.axisColor)
        ax[1,dir].add_patch(Rectangle((rectX_wind, rectY_wind), rectWid_wind, rectHeight_wind,
            facecolor=const.axisColor))
        ax[1,dir].text(rectX_wind+rectWid_wind/2, rectY_wind-rectHeight_wind*3-const.fontSize_stimBar/const.scaleAng_rawTraces,
            str(2)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='center',
            color=const.axisColor)

        # draw scale bar
        if dir == 0:
            rectHeight = 5
            rectX = 285#
            rectY = 10#
            rectWid = const.scaleAng_width
            ax[0,0].add_patch(Rectangle((rectX,rectY),rectWid,rectHeight,facecolor = const.axisColor))
            ax[0,0].text(rectX+10, rectY+rectHeight/2, str(rectHeight)+' deg',
                fontsize=const.fontSize_stimBar,horizontalalignment='left',color=const.axisColor)
            ax[0,0].text(0,23,'2nd',fontsize=const.fontSize_stimBar,horizontalalignment='left',color=const.axisColor)
            ax[1,0].text(0,23,'3rd',fontsize=const.fontSize_stimBar,horizontalalignment='left',color=const.axisColor)

    # plot 3rd and 2nd displacements versus baseline

    # Split the data into training/testing sets
    rightDeflect2_X_train = rightBase2#
    rightDeflect2_X_train = rightDeflect2_X_train.reshape(-1, 1)
    rightDeflect2_X_test = rightBase2#
    rightDeflect2_X_test = rightDeflect2_X_test.reshape(-1, 1)
    # Split the targets into training/testing sets
    rightDeflect2_y_train = rightDeflect2#
    rightDeflect2_y_train = rightDeflect2_y_train.reshape(-1, 1)
    rightDeflect2_y_test = rightDeflect2#
    rightDeflect2_y_test = rightDeflect2_y_test.reshape(-1, 1)
    regr = linear_model.LinearRegression() # Create linear regression object
    regr.fit(rightDeflect2_X_train, rightDeflect2_y_train) # Train the model using the training sets
    rightDeflect2_y_pred = regr.predict(rightDeflect2_X_test) # Make predictions using the testing set
    print("Coefficients(r2): \n", regr.coef_) # The coefficients
    print("Mean squared error(r2): %.2f" % mean_squared_error(rightDeflect2_y_test, rightDeflect2_y_pred)) # The mean squared error
    print("Coefficient of determination(r2): %.2f" % r2_score(rightDeflect2_y_test, rightDeflect2_y_pred)) # The coefficient of determination: 1 is perfect prediction
    # Plot outputs
    ax2[0,0].plot(rightBase2,rightDeflect2,'.',color='black')
    ax2[0,0].plot(rightDeflect2_X_test, rightDeflect2_y_pred, color="blue", linewidth=2)
    ax2[0,0].set_title('2nd segment, r2=\n' + str(r2_score(rightDeflect2_y_test, rightDeflect2_y_pred)))
    # Split the data into training/testing sets
    rightDeflect3_X_train = rightBase3#
    rightDeflect3_X_train = rightDeflect3_X_train.reshape(-1, 1)
    rightDeflect3_X_test = rightBase3#
    rightDeflect3_X_test = rightDeflect3_X_test.reshape(-1, 1)
    # Split the targets into training/testing sets
    rightDeflect3_y_train = rightDeflect3#
    rightDeflect3_y_train = rightDeflect3_y_train.reshape(-1, 1)
    rightDeflect3_y_test = rightDeflect3#
    rightDeflect3_y_test = rightDeflect3_y_test.reshape(-1, 1)
    regr = linear_model.LinearRegression() # Create linear regression object
    regr.fit(rightDeflect3_X_train, rightDeflect3_y_train) # Train the model using the training sets
    rightDeflect3_y_pred = regr.predict(rightDeflect3_X_test) # Make predictions using the testing set
    print("Coefficients(r3): \n", regr.coef_) # The coefficients
    print("Mean squared error(r3): %.2f" % mean_squared_error(rightDeflect3_y_test, rightDeflect3_y_pred)) # The mean squared error
    print("Coefficient of determination (r3): %.2f" % r2_score(rightDeflect3_y_test, rightDeflect3_y_pred)) # The coefficient of determination: 1 is perfect prediction
    ax2[0,1].plot(rightBase3,rightDeflect3,'.',color='black')
    ax2[0,1].plot(rightDeflect3_X_test, rightDeflect3_y_pred, color="blue", linewidth=2)
    ax2[0,1].set_title('3rd segment, r2=\n' + str(r2_score(rightDeflect2_y_test, rightDeflect2_y_pred)))

    leftDeflect2_X_train = leftBase2#
    leftDeflect2_X_train = leftDeflect2_X_train.reshape(-1, 1)
    leftDeflect2_X_test = leftBase2#
    leftDeflect2_X_test = leftDeflect2_X_test.reshape(-1, 1)
    # Split the targets into training/testing sets
    leftDeflect2_y_train = leftDeflect2#
    leftDeflect2_y_train = leftDeflect2_y_train.reshape(-1, 1)
    leftDeflect2_y_test = leftDeflect2#
    leftDeflect2_y_test = leftDeflect2_y_test.reshape(-1, 1)
    regr = linear_model.LinearRegression() # Create linear regression object
    regr.fit(leftDeflect2_X_train, leftDeflect2_y_train) # Train the model using the training sets
    leftDeflect2_y_pred = regr.predict(leftDeflect2_X_test) # Make predictions using the testing set
    print("Coefficients(r2): \n", regr.coef_) # The coefficients
    print("Mean squared error(r2): %.2f" % mean_squared_error(leftDeflect2_y_test, leftDeflect2_y_pred)) # The mean squared error
    print("Coefficient of determination(r2): %.2f" % r2_score(leftDeflect2_y_test, leftDeflect2_y_pred)) # The coefficient of determination: 1 is perfect prediction

    ax2[1,0].plot(leftBase2,leftDeflect2,'.',color='black')
    ax2[1,0].plot(leftDeflect2_X_test, leftDeflect2_y_pred, color="blue", linewidth=2)
    ax2[1,0].set_title('2nd segment, r2=\n' + str(r2_score(leftDeflect2_y_test, leftDeflect2_y_pred)))

    leftDeflect3_X_train = leftBase3#
    leftDeflect3_X_train = leftDeflect3_X_train.reshape(-1, 1)
    leftDeflect3_X_test = leftBase3#
    leftDeflect3_X_test = leftDeflect3_X_test.reshape(-1, 1)
    # Split the targets into training/testing sets
    leftDeflect3_y_train = leftDeflect3#
    leftDeflect3_y_train = leftDeflect3_y_train.reshape(-1, 1)
    leftDeflect3_y_test = leftDeflect3#
    leftDeflect3_y_test = leftDeflect3_y_test.reshape(-1, 1)
    regr = linear_model.LinearRegression() # Create linear regression object
    regr.fit(leftDeflect3_X_train, leftDeflect3_y_train) # Train the model using the training sets
    leftDeflect3_y_pred = regr.predict(leftDeflect3_X_test) # Make predictions using the testing set
    print("Coefficients(r2): \n", regr.coef_) # The coefficients
    print("Mean squared error(r2): %.2f" % mean_squared_error(leftDeflect3_y_test, leftDeflect3_y_pred)) # The mean squared error
    print("Coefficient of determination(r2): %.2f" % r2_score(leftDeflect3_y_test, leftDeflect3_y_pred)) # The coefficient of determination: 1 is perfect prediction
    ax2[1,1].plot(leftBase3,leftDeflect3,'.',color='black')
    ax2[1,1].plot(leftDeflect3_X_test, leftDeflect3_y_pred, color="blue", linewidth=2)
    ax2[1,1].set_title('3rd segment, r2=\n' + str(r2_score(leftDeflect2_y_test, leftDeflect2_y_pred)))

    # 3rd vs. 2nd plot configuration
    ax[0,5].set_ylim([-25, 25])
    ax[0,5].set_xlim([-25, 25])
    ax[0,5].set_ylabel('second seg')
    ax[0,5].set_xlabel('third seg')

    ax2[0,0].set_xlim([-45,-20])
    ax2[1,0].set_xlim([-45,-20])
    ax2[0,1].set_xlim([54,75])
    ax2[1,1].set_xlim([54,75])
    ax2[0,0].set_ylabel('right antenna')
    ax2[1,0].set_ylabel('left antenna')
    plt.setp(ax2, xlabel='baseline', ylim=[-8,18])

    # Plot average second segment deflection across all directions
    avgTraceAll = np.nanmean(traces_2_all[:,rangeTrace],0)
    avgTraceAll = avgTraceAll-np.nanmean(avgTraceAll[0:45])
    numTraces = np.shape(traces_2_all)[1]
    stderr = np.nanstd(traces_2_all[:,rangeTrace],0)/np.sqrt(numTraces) # std/sqrt(#flies)= standard error
    #plot cross-fly data
    xx = np.linspace(0,np.shape(avgTraceAll)[0],np.shape(avgTraceAll)[0])
    ax[1,5].fill_between(xx,avgTraceAll-stderr, avgTraceAll+stderr,
        edgecolor=None,facecolor=const.color_inactivationNoLight,alpha=const.transparencyPatch,interpolate=True)

    ax[1,5].plot(avgTraceAll,color=const.dark_gray)
    ax[1,5].set_title('gray:stderr with n='+str(numTraces) +' traces')
    # configure cross-direction second segment avg plot
    ax[1,5].set_facecolor(const.figColor)
    ax[1,5].axis('off')
    ax[1,5].set_facecolor(const.figColor)
    ax[1,5].set_ylim([-8, 8])
    ax[1,5].set_xlim([-5, 290])
    # draw wind stimulus bar
    rectX = 285
    rectY = 2
    stimSize = 3
    ax[1,5].add_patch(Rectangle((rectX, rectY), rectWid, stimSize,
        facecolor=const.axisColor))
    ax[1,5].text(rectX+rectWid/2, rectY-rectHeight*3-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(2)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='center',
        color=const.axisColor)
    ax[1,5].text(rectX+10, rectY+rectHeight/2, str(stimSize)+' deg',
        fontsize=const.fontSize_stimBar,horizontalalignment='left',color=const.axisColor)

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig:  # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        # save second and third segment traces
        savepath_png = figPath + 'secondThirdDisplacement_CS_activate.png'
        savepath_pdf = figPath + 'secondThirdDisplacement_CS_activate.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())
        # save baseline vs. wind deflection
        savepath_png = figPath + 'baselineVsDisplacment_CS_activate.png'
        savepath_pdf = figPath + 'baselineVsDisplacment_CS_activate.pdf'
        fig2.savefig(savepath_png, facecolor=fig2.get_facecolor())
        fig2.savefig(savepath_pdf, facecolor=fig2.get_facecolor())


# plot displacement of the second antennal segment during wind, for all directions, for each speed (200,100,50,0 cm/s)
# For Canton-S (activation) control flies
def plot_second_displacement_velocity(importAnew = 0,direction=2,savefig = 0,plotIndvTraces=0):
    savematfiles = 1 #if 1, will save the traces with direction embedded in name (e.g. for classifying fast vs. slow movements!)
    directions = [0,1,2,3,4]
    shiftForLight = 70
    rangeTrace = range(shiftForLight,350,1) #area around wind stimulus (all during light)
    windOn = 120-shiftForLight #

    # Plot the data
    fig, ax = plt.subplots(1,1,facecolor=const.figColor, figsize=(8, 8))

    fn_second200 = const.savedDataDirectory+'traces_CS_activate_200_secondSegment_'+str(direction)+'.npy'
    fn_second100 = const.savedDataDirectory+'traces_CS_activate_100_secondSegment_'+str(direction)+'.npy'
    fn_second50 = const.savedDataDirectory+'traces_CS_activate_50_secondSegment_'+str(direction)+'.npy'
    fn_second0 = const.savedDataDirectory+'traces_CS_activate_0_secondSegment_'+str(direction)+'.npy'
    if ~importAnew & os.path.isfile(fn_second200):
        traces_second200 = np.load(fn_second200)
        traces_second100 = np.load(fn_second100)
        traces_second50 = np.load(fn_second50)
        traces_second0 = np.load(fn_second0)
    else:
        traces_second200 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[0,2],speed=3,direction=direction)#=-1)
        np.save(fn_second200, traces_second200)
        traces_second100 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[0,2],speed=2,direction=direction)#=-1)
        np.save(fn_second100, traces_second100)
        traces_second50 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[0,2],speed=1,direction=direction)#=-1)
        np.save(fn_second50, traces_second50)
        traces_second0 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[0,2],speed=0,direction=direction)#=-1)
        np.save(fn_second0, traces_second0)
        print('Importing traces:\n'+fn_second200)

    if savematfiles == 1:
        filesToConvert = [fn_second200, fn_second100, fn_second50, fn_second0]
        for ind, fn in enumerate(filesToConvert):
            print(ind)
            print(fn[:-4])
            data = np.load(fn)
            mat_fn = fn[:-4]+'.mat'
            savemat(mat_fn,{'data':data})

    #Print sample sizes:
    print('0 cm/s: ')
    print( np.shape(traces_second0))
    print('50 cm/s: ')
    print(np.shape(traces_second50))
    print('100 cm/s: ')
    print( np.shape(traces_second100))
    print('200 cm/s: ')
    print( np.shape(traces_second200))

    #plot average deflection
    avgTrace200 = np.nanmean(traces_second200[:,rangeTrace],0)
    avgTrace200 = avgTrace200-np.nanmean(avgTrace200[0:45])
    secondSegError200 = np.nanstd(traces_second200[:,rangeTrace],0)/np.sqrt(np.shape(traces_second200)[0]) #stderr
    xx = np.linspace(0,np.shape(rangeTrace)[0]-1,np.shape(rangeTrace)[0])
    if plotIndvTraces == 1:
        for tt in range(np.shape(traces_second200)[0]):
            trace = traces_second200[tt,rangeTrace]
            ax.plot(trace-np.nanmean(trace[0:40]),color=const.colors_velocity_RminL[0],linewidth=const.traceWidAvg,alpha=const.transparencyPatch)
    else:
        ax.fill_between(xx,avgTrace200-secondSegError200,
            avgTrace200+secondSegError200,
            edgecolor=None,facecolor=const.colors_velocity_RminL[0],alpha=const.transparencyPatch)
    ax.plot(avgTrace200,color=const.colors_velocity_RminL[0],linewidth=const.traceWidAvg)

    avgTrace100 = np.nanmean(traces_second100[:,rangeTrace],0)
    avgTrace100 = avgTrace100-np.nanmean(avgTrace100[0:45])
    secondSegError100 = np.nanstd(traces_second100[:,rangeTrace],0)/np.sqrt(np.shape(traces_second100)[0]) #stderr
    xx = np.linspace(0,np.shape(rangeTrace)[0]-1,np.shape(rangeTrace)[0])
    if plotIndvTraces == 1:
        for tt in range(np.shape(traces_second100)[0]):
            trace = traces_second100[tt,rangeTrace]
            ax.plot(trace-np.nanmean(trace[0:40]),color=const.colors_velocity_RminL[1],linewidth=const.traceWidAvg,alpha=const.transparencyPatch)
    else:
        ax.fill_between(xx,avgTrace100-secondSegError100,
            avgTrace100+secondSegError100,
            edgecolor=None,facecolor=const.colors_velocity_RminL[1],alpha=const.transparencyPatch)
    ax.plot(avgTrace100,color=const.colors_velocity_RminL[1],linewidth=const.traceWidAvg)

    avgTrace50 = np.nanmean(traces_second50[:,rangeTrace],0)
    avgTrace50 = avgTrace50-np.nanmean(avgTrace50[0:45])
    secondSegError50 = np.nanstd(traces_second50[:,rangeTrace],0)/np.sqrt(np.shape(traces_second50)[0]) #stderr
    xx = np.linspace(0,np.shape(rangeTrace)[0]-1,np.shape(rangeTrace)[0])
    if plotIndvTraces == 1:
        for tt in range(np.shape(traces_second50)[0]):
            trace = traces_second50[tt,rangeTrace]
            ax.plot(trace-np.nanmean(trace[0:40]),color=const.colors_velocity_RminL[2],linewidth=const.traceWidAvg,alpha=const.transparencyPatch)
    else:
        ax.fill_between(xx,avgTrace50-secondSegError50,
            avgTrace50+secondSegError50,
            edgecolor=None,facecolor=const.colors_velocity_RminL[2],alpha=const.transparencyPatch)
    ax.plot(avgTrace50,color=const.colors_velocity_RminL[2],linewidth=const.traceWidAvg)

    avgTrace0 = np.nanmean(traces_second0[:,rangeTrace],0)
    avgTrace0 = avgTrace0-np.nanmean(avgTrace0[0:45])
    secondSegError0 = np.nanstd(traces_second0[:,rangeTrace],0)/np.sqrt(np.shape(traces_second0)[0]) #stderr
    xx = np.linspace(0,np.shape(rangeTrace)[0]-1,np.shape(rangeTrace)[0])
    if plotIndvTraces == 1:
        for tt in range(np.shape(traces_second0)[0]):
            trace = traces_second0[tt,rangeTrace]
            ax.plot(trace-np.nanmean(trace[0:40]),color=const.colors_velocity_RminL[3],linewidth=const.traceWidAvg,alpha=const.transparencyPatch)
    else:
        ax.fill_between(xx,avgTrace0-secondSegError0,
            avgTrace0+secondSegError0,
            edgecolor=None,facecolor=const.colors_velocity_RminL[3],alpha=const.transparencyPatch)
    ax.plot(avgTrace0,color=const.colors_velocity_RminL[3],linewidth=const.traceWidAvg)

    # draw wind stimulus bar
    #stimulus bar constants
    rectHeight_wind = 0.03
    rectX_wind = 45  # -1 to align with 0 indexing
    rectY_wind = -0.15
    rectWid_wind = int(2*const.framerate)
    ax.add_patch(Rectangle((rectX_wind, rectY_wind), rectWid_wind, rectHeight_wind,
        facecolor=const.axisColor))
    ax.text(rectX_wind+rectWid_wind/2, rectY_wind-rectHeight_wind*3-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(2)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='center',
        color=const.axisColor)

    # draw scale bar
    rectHeight = 1
    rectX = -1#
    rectY = 1#
    rectWid = 2
    ax.add_patch(Rectangle((rectX,rectY),rectWid,rectHeight,facecolor = const.axisColor))
    ax.text(rectX+10, rectY+rectHeight/2, str(rectHeight)+' deg',
        fontsize=const.fontSize_stimBar,horizontalalignment='left',color=const.axisColor)
    ax.text(0,23,'2nd',fontsize=const.fontSize_stimBar,horizontalalignment='left',color=const.axisColor)
    ax.text(0,23,'3rd',fontsize=const.fontSize_stimBar,horizontalalignment='left',color=const.axisColor)

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig:  # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        # save second and third segment traces
        savepath_png = figPath + 'secondDisplacement_velocity_CS_activate_dir'+str(direction)+'.png'
        savepath_pdf = figPath + 'secondDisplacement_velocity_CS_activate_dir'+str(direction)+'.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())

# plot displacement of the second antennal segment during wind, for all directions, for each speed (200,100,50,0 cm/s)
# For Canton-S (activation) control flies
def plot_second_vs_third_200cm(dir=0,importAnew = 0,savefig = 0):

    sp = 3 #200 cm/s
    shiftForLight = 70
    rangeTrace = range(shiftForLight,350,1) #area around wind stimulus (all during light)

    # Plot the data
    fig, ax = plt.subplots(1,1,facecolor=const.figColor, figsize=(8, 8))

    fn_200_second = const.savedDataDirectory+'traces_CS_activate_200_second_'+str(dir)+'.npy'
    fn_200_third = const.savedDataDirectory+'traces_CS_activate_200_third_'+str(dir)+'.npy'
    if ~importAnew & os.path.isfile(fn_200_second):
        traces_second200 = np.load(fn_200_second)
        traces_third200 = np.load(fn_200_third)
    else:
        traces_second200 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[0],speed=sp,direction=dir)
        np.save(fn_200_second, traces_second200)
        traces_third200 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[1],speed=sp,direction=dir)
        np.save(fn_200_third, traces_third200)
        print('Importing traces:\n'+fn_200_second)

    #plot average deflections versus each other - from within wind region
    windZoomOn = 45
    windZoomOff = 60
    timeZoom = (windZoomOff-windZoomOn)/60 #sec
    zoomWind = range(windZoomOn,windZoomOff,1) #region at onset
    avg200second = traces_second200[:,rangeTrace]
    avg200second = avg200second-np.nanmean(avg200second[0:45])
    avg200third = traces_third200[:,rangeTrace]
    avg200third = avg200third-np.nanmean(avg200third[0:45])
    ax.plot(avg200third[:,zoomWind],avg200second[:,zoomWind],'.',color='black',linewidth=const.traceWidAvg)

    # Plot least squares line
    X_train = avg200third[~np.isnan(avg200third)&~np.isnan(avg200second)]
    X_train = X_train.reshape(-1, 1)
    X_test = avg200third[~np.isnan(avg200third)&~np.isnan(avg200second)]
    X_test = X_test.reshape(-1, 1)
    y_train = avg200second[~np.isnan(avg200third)&~np.isnan(avg200second)]
    y_train = y_train.reshape(-1, 1)
    y_test = avg200second[~np.isnan(avg200third)&~np.isnan(avg200second)]
    y_test = y_test.reshape(-1, 1)

    regr = linear_model.LinearRegression() # Create linear regression object
    regr.fit(X_train, y_train) # Train the model using the training sets
    y_pred = regr.predict(X_test) # Make predictions using the testing set
    print("Coefficients(r2): \n", regr.coef_) # The coefficients
    print('Slope: \n', regr.coef_[0])
    print('Intercept: \n', regr.intercept_)
    print("Mean squared error(r2): %.2f" % mean_squared_error(y_test, y_pred)) # The mean squared error
    print("Coefficient of determination(r2): %.2f" % r2_score(y_test, y_pred)) # The coefficient of determination: 1 is perfect prediction
    # Plot outputs
    ax.plot(X_test, y_pred, color="blue", linewidth=2)
    ax.plot(range(15,-15,-1), range(-15,15,1), color='black', linewidth=2, linestyle='--')
    ax.set_xlabel('third segment (deg)')
    ax.set_ylabel('second segment (deg)')
    ax.set_title('third vs. second segment\n'+const.windDirLabels_noSymbols[dir]+'\n('+str(timeZoom)+' s wind onset)\n '
        + 'rsq='+str(r2_score(y_test, y_pred)))

    axlen = 40
    ax.set_ylim([-15, 15])
    ax.set_xlim([-15, 15])

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig:  # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        # save second and third segment traces
        savepath_png = figPath + 'third_vs_second_200cm_neg45_CS_activate_'+const.windDirLabels_noSymbols[dir]+'.png'
        savepath_pdf = figPath + 'third_vs_second_200cm_neg45_CS_activate_'+const.windDirLabels_noSymbols[dir]+'.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())

# plot initial displacement of the third antennal segment during wind vs. # active movements during that wind stimulus
# for all directions, for each speed (200,100,50,0 cm/s)
# For Canton-S (activation) control flies
def plot_third_vs_active_count(dir=0,importAnew = 0,savefig = 0):

    sp = 3 #200 cm/s
    rangeOnset = range(122,130,1) #short region just after wind onset
    rangeBaseline = list(range(0, 59))#baseline inds
    # Plot the data
    fig, ax = plt.subplots(1,1,facecolor=const.figColor, figsize=(8, 8))
    fn_200_third = const.savedDataDirectory+'traces_CS_activate_200_third_'+str(dir)+'.npy'
    if ~importAnew & os.path.isfile(fn_200_third):
        traces_third200 = np.load(fn_200_third)
    else:
        traces_third200 = get_all_raw_traces(expt='CS_activate',cameraView='frontal',allowFlight = 1,inactivate=0,ANGS=[1],speed=sp,direction=dir)
        np.save(fn_200_third, traces_third200)
        print('Importing traces:\n'+fn_200_second)

    #plot average deflections versus each other - from within wind region
    windZoomOn = 45
    windZoomOff = 60
    timeZoom = (windZoomOff-windZoomOn)/60 #sec
    zoomWind = range(windZoomOn,windZoomOff,1) #region at onset
    onsetDeflection = traces_third200[:,rangeOnset]
    baseline = traces_third200[:,rangeBaseline]
    onsetDeflection = np.nanmean(onsetDeflection,1)-np.nanmean(baseline,1) #baseline-subtract wind deflection

    ax.plot(onsetDeflection,'.',color='black',linewidth=const.traceWidAvg) #plot wind deflection vs. # active movements during this trial
    axlen = 40

    plt.pause(0.001)
    plt.show()
    plt.show(block=False)

    if savefig:  # save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        print('Saving figure here: ' + figPath)

        # save second and third segment traces
        savepath_png = figPath + 'third_vs_second_200cm_neg45_CS_activate_'+const.windDirLabels_noSymbols[dir]+'.png'
        savepath_pdf = figPath + 'third_vs_second_200cm_neg45_CS_activate_'+const.windDirLabels_noSymbols[dir]+'.pdf'
        fig.savefig(savepath_png, facecolor=fig.get_facecolor())
        fig.savefig(savepath_pdf, facecolor=fig.get_facecolor())


# configure axes for autocorr average comparison
def set_acorr_axes(ax,maxlag,ylim,ybounds,xticklabels):

    ax.set_facecolor(const.figColor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(const.axisColor)
    ax.spines['bottom'].set_color(const.axisColor)
    ax.tick_params(direction='in', length=5, width=0.5)
    ax.tick_params(axis='y',colors=const.axisColor)
    #configure the y-axis
    ax.spines['left'].set_bounds(ybounds[0],ybounds[1]) #do not extend y-axis line beyond ticks
    ax.set_ylim(ylim[0], ylim[1])
    #configure the x-axis
    ax.set_xticks([0, maxlag, maxlag*2])
    ax.spines['bottom'].set_bounds(0,maxlag*2) #do not extend x-axis line beyond ticks
    ax.set_xticklabels(xticklabels)#['-0.5','0','0.5'])
    ax.set_xlabel('lag (s)')
    ax.tick_params(axis='x',colors=const.axisColor)

# Generate butterworth filter
def generate_butter_filter(samplerate, fc = 10):
    ww = fc/(samplerate/2) #normalize the frequency
    bb, aa = signal.butter(4, ww, 'low')
    return bb, aa

# Use raw tachometer signal to detect when fly is flying (reflected IR light off wings)
def detect_flight(tach, bb, aa, TEST = 0):
    tach = np.concatenate(tach,axis=0)
    endTrial = 8*const.samplerate
    timeWindowWingbeats = 400 #number of samples during which minimum # wingbeats must occur to detect start of flight

    minPreFlightSamples = 75 #number of samples before flight start to quality as 'start' (excludes trials in which the fly is continuously flying!)
    #thresholds for detecting peaks of wingbeat
    thresh_amp = 1#1.5#0.75#1.5 #voltage over which we consider tachometer signal to indicate flight
    thresh_distance = 33 #requires two samples to represent no more than 300 Hz (more than maximal qingbeat frequency)
    thresh_wid_max = 60 #exclude large-width signal (not wingbeats, usually wiggling!)
    thresh_wid_min = 5

    nn = np.shape(tach)[0]
    tach_filtered = np.zeros(nn)
    tach_filtered2 = np.zeros(nn)
    tach_filtered3 = np.zeros(nn)
    kk = 15 #number of points to average over
    for ii in range(kk,nn-kk-1):
        tach_filtered[ii] = np.mean(tach[ii-kk:ii+kk])
    kk = 30 #number of points to average over
    for ii in range(kk,nn-kk-1):
        tach_filtered2[ii] = np.mean(tach[ii-kk:ii+kk])
    kk = 50 #number of points to average over
    for ii in range(kk,nn-kk-1):
        tach_filtered3[ii] = np.mean(tach[ii-kk:ii+kk])

    #add buffer to start and end of signal to avoid warping after filtering
    numBufSamples = 100 #number of samples to wrap beginning and end of trial (avoid counting continuous slight at start/end as start/stop flights)
    tachWithBuffer = np.concatenate([tach[0:numBufSamples], tach, tach[-numBufSamples:]])
    tachFilt = signal.filtfilt(bb,aa,abs(tachWithBuffer)) #low-pass signal
    tachFilt = tachFilt[numBufSamples:-numBufSamples] #trim the buffer off
    isAboveThresh = tachFilt>thresh_amp #find inds over threshold
    tachOverThresh = np.array([ii for ii, xx in enumerate(isAboveThresh) if xx])

    magDiff = 1000 #expand trace so we can see it with other signals
    diffThresh = 0.002*magDiff  #threshold over which we'll consider flight starts
    diffTach = np.diff(tachFilt)*magDiff
    diffPeaks, _ = signal.find_peaks(diffTach, height=diffThresh)

    #for WBF, find peaks of raw tachometer signal
    peaks, _ = signal.find_peaks(tach, height=thresh_amp,
        distance=thresh_distance, width=[thresh_wid_min,thresh_wid_max])

    #Determine if peaks represent flight starts, a stop,
    #  continuous flight, or blips to be ignored!
    if (tachOverThresh.size != 0) & (np.shape(tachOverThresh)[0] >= timeWindowWingbeats):
        #check if minimum # wingbeats, and the last wingbeat falls within the time window (if the flight lasts long enough to count)
        #this is a trial in which the fly just starts flying (or short start then stop)
        if (tachOverThresh[timeWindowWingbeats-1] < (tachOverThresh[0]+timeWindowWingbeats)) & (tachOverThresh[0] > minPreFlightSamples):
            flightStart = tachOverThresh[0]
        else: flightStart = []
        #trials in which the fly is flying with one flight stop (or short start then stop):
        if (tachOverThresh[-1] > tachOverThresh[-timeWindowWingbeats]) and ((endTrial-tachOverThresh[-1]) > timeWindowWingbeats):
            flightStop = tachOverThresh[-1]
        else: flightStop = []
        #final case: the fly is flying and stops then restarts during trial
    else:
        flightStart = []
        flightStop = []
    #verify that there are wingbeats within the flight region (if not, this is probably just a leg wiggling blip)
    if flightStart and (np.size(peaks) == 0): flightStart=[]
    if flightStop and (np.size(peaks) == 0): flightStop=[]

    if TEST:
        fig, ax = plt.subplots(2,1,facecolor=[1,1,1], figsize=(10,8))
        ax[0].plot(abs(tach), color='gray')
        ax[0].plot(tachFilt, color='blue')
        if tachOverThresh.size != 0:
            ax[0].plot(tachOverThresh, tachFilt[tachOverThresh], '.' , color='red') #plot region classified as flight!
        #plot raw tachometer signal for wingbeat frequency analysis
        ax[1].plot(tach, color='gray')
        ax[1].plot(peaks, tach[peaks], 'x', color='green')

        plt.suptitle('Tachometer signal classification (red indicates flight)', color='black')
        ax[1].set_ylim(-10, 10)
        plt.show(block=False)
        plt.pause(0.001)
        plt.show()
        return tachOverThresh, fig, ax

    return tachOverThresh, isAboveThresh

# Function to look at flight classification in all single trials from one fly
# Will display raw tachometer signal, filtered version, and region categorized as flight
# Plots single trials one by one
def plot_flight_classification(expt = '2020_12_14_E1'):
    import loadNotes_mnOptoVelocityTuning as ln
    # plot across all flies for a given experiment
    try:
        exptNotes = get_mat_notes(expt)
    except:
        print(expt)
        print('Please check the name of the experiment and try again!')
        return

    for trialNum in range(np.shape(exptNotes)[0]):
        print('trial number '+str(trialNum+1))
        tach = exptNotes.tachometer[trialNum]
        bb, aa = generate_butter_filter(samplerate = const.samplerate) #get the butterworth filter
        _, fig, ax = detect_flight(tach,bb,aa,1) #detech flight
        plt.suptitle(expt + ' trial ' + str(trialNum+1) + ' tachometer signal classification')
        plt.show(block=True)

# Plot raw antennal angles for a given trial (only some antenna angles)
# All angles plotted as relative to midline axis, with second segment averaged
#  as listed in constants_activePassiveAnalysis
def plot_antenna_angles(expt='2020_11_13_E4', trialNum=15, cameraView='frontal', plotTach = 0, savefig=0):

    angles = get_antenna_angles_adjusted(expt, cameraView, trialNum)
    fig, axAng = plt.subplots(2,1,facecolor=[1,1,1], figsize=(8,8),gridspec_kw={'height_ratios':[8,1]})

    font = FontProperties() #set font
    font.set_family(const.fontFamily)
    font.set_name(const.fontName)
    # grab notes associated with this experiment
    exptNotes = get_mat_notes(expt)
    tach = exptNotes.tachometer[trialNum-1]

    stimTime = exptNotes.trial_time_wind[0][0]
    stimStart = exptNotes.pre_trial_time[0][0]
    stimStop = exptNotes.pre_trial_time[0][0]+stimTime
    framerate = exptNotes.fps[0][0]
    numPreInds = int(exptNotes.pre_trial_time[0]*framerate)
    baselineInds = list(range(0, numPreInds))
    optoStimTime = const.activateTime
    # plot wind direction and velocity
    valveState = exptNotes.valveState[trialNum-1]
    windDir = const.windDirections[valveState[0]-1]

    axAng[0].text(angles.shape[0]+5, angles.shape[1]*const.shiftYTraces,
        windDir+' deg wind', fontsize=const.fontSize_angPair, color=const.axisColor)
    if cameraView == "frontal":
        velocity = exptNotes.velocity[trialNum-1][0]
        axAng[0].text(angles.shape[0]+5, angles.shape[1]*const.shiftYTraces*1.05,
            str(velocity)+' cm/s', fontsize=const.fontSize_angPair, color = const.axisColor)

    for idx in range(angles.shape[1]):
        baseline = np.nanmean(angles[baselineInds, idx])
        shift = idx*const.shiftYTraces  # shifts traces relative to each other (ease of viewing)
        axAng[0].plot(angles[:, idx]-baseline+shift, color=const.colors_antAngs[idx])
        axAng[0].text(angles.shape[0]+5, shift, const.angPairNames[cameraView][idx],
            fontsize=const.fontSize_angPair, color = const.colors_antAngs[idx])

    # draw stimulus bars: wind and optogenetic activation periods, then scale bar
    ymin = np.nanmin(angles[:, 0])-np.nanmean(angles[baselineInds, 0]) #minimum of first trace plotted
    rectHeight = const.stimBar_height

    # draw wind stimulus bar
    rectX = int(stimStart*framerate)-1  # -1 to align with 0 indexing
    rectY = ymin-rectHeight*2
    rectWid = int(stimStop*framerate-stimStart*framerate)
    axAng[0].add_patch(Rectangle((rectX, rectY), rectWid, rectHeight,
        facecolor=const.axisColor))
    axAng[0].text(rectX+rectWid/2, rectY-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(stimTime)+' s wind',fontsize=const.fontSize_stimBar,horizontalalignment='center',
        color=const.axisColor)
    axAng[0].set_ylim(ymin-rectHeight*3, 10*(angles.shape[1]+1))
    # draw light activation stimulus bar (if opto!)
    rectX = int(const.activateStart*framerate)-1  # -1 to align with 0 indexing
    rectY = ymin-rectHeight*2-rectHeight*4  # add some more to accomodate wind stimulus (above)
    rectWid = int(const.activateTime*framerate)
    axAng[0].add_patch(Rectangle((rectX, rectY), rectWid, rectHeight,
        facecolor=const.color_activateColor))
    axAng[0].text(rectX+rectWid/2, rectY-rectHeight-const.fontSize_stimBar/const.scaleAng_rawTraces,
        str(optoStimTime)+' s light',fontsize=const.fontSize_stimBar,horizontalalignment='center',
        color=const.color_activateColor)

    # draw scale bar
    rectHeight = const.scaleAng_rawTraces
    rectX = const.scaleAng_xLocation
    rectY = const.shiftYTraces
    rectWid = const.scaleAng_width
    axAng[0].add_patch(Rectangle((rectX,rectY),rectWid,rectHeight,facecolor = const.axisColor))
    axAng[0].text(rectX-rectWid/2, rectY+rectHeight/2, str(const.scaleAng_rawTraces)+' deg',
        fontsize=const.fontSize_stimBar,horizontalalignment='right',color=const.axisColor)

    axAng[0].set_ylim(ymin-rectHeight*1, 10*(angles.shape[1]+1))
    axAng[0].set_facecolor([1, 1, 1])
    axAng[0].axis('off')

    if plotTach: #plot raw tachometer signal below the antenna traces (indicating flight/no flight)
        axAng[1].plot(tach, color='gray')
        axAng[1].axis('off')
        axAng[1].set_ylim(-10, 10)
        bb, aa = generate_butter_filter(samplerate = const.samplerate)
        tachOverThresh, overThresh = detect_flight(tach, bb, aa)
        axAng[1].plot(tachOverThresh, tach[tachOverThresh], color='red')

        windStart =stimStart*const.samplerate
        windStop = stimStop*const.samplerate
        isInWindRegion = 1*np.array([(tachOverThresh>=windStart) & (tachOverThresh<=windStop)])
        tachOverThreshPercent = np.sum(isInWindRegion)/(windStop-windStart)
        axAng[1].text(tach.shape[0]+1000, 0, 'tachometer',
            fontsize=const.fontSize_angPair, color = 'gray')
        print(tachOverThreshPercent)

        isFlyingDuringWind = (tachOverThreshPercent >= const.percentageFlying)
        if isFlyingDuringWind:
            print('Flying trial! <-- in plot_antenna_angles ~line 1128')
            print('Percentage of flight during wind stimulus: '+str(tachOverThreshPercent))
            axAng[1].text(tach.shape[0]+1000, -10, 'flight trial',
                fontsize=const.fontSize_angPair, color = 'gray')
    else: isFlyingDuringWind = 0

    plt.suptitle('Angles of first, second antennal segments \n '+expt
        +' - ' + cameraView + ' mount, trial #' + str(trialNum), color='black')
    print(trialNum)

    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    if savefig: #save in folder with today's date
        today = date.today()
        dateStr = today.strftime("%Y_%m_%d")
        figPath = const.savedFigureDirectory+str(dateStr)+'/'
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
        savepath = figPath+'tracesSingleExpt_'+expt+'_trial_'+str(trialNum)+ '_'+ cameraView + '.png'
        print('Saving figure here: ' + savepath)
        plt.savefig(savepath, transparent=True)

    return isFlyingDuringWind

# a small random-number generator for introducing jitter into 1-D data
def jitter(len):
    from numpy.random import rand
    jitter = np.ones(len) # *maxVal
    jitter = jitter+rand(len)-1
