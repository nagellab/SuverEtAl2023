"""
SuverEtAl_2023_constants.py

Constants associated with analysis and plotting for Suver et al. 2023

"""

import numpy as np

notesName = 'SuverEtAl2023_loadNotes'
baseDirectory = '..'
matDirectory = '../SuverEtAl2023_Data/MatFiles_Notes/'
savedDataDirectory = baseDirectory+'/SuverEtAl2023_Data/SavedAngles/'
savedFigureDirectory = baseDirectory+'/SuverEtAl2023_Data/Plots/'
savedDataDirectoryForSingleTrials = baseDirectory+'/SuverEtAl2023_Data/SavedSingleTrialData_2022_12_01/'

percentageFlying = 1
trialThreshForQuant = 2 # threshold (# trials) for inclusion in quantification of active movements

THRESH_ANG_CUTOFF = 35 #baseline-subracted antenna angle over which antenna never reaches normally (tracking error!)
TRACKING_ERROR_SIZE_CUTOFF = 35 #

colormap = 'jet' #matches tracking via DLC

windDirections = ['-90','-90','-45','-45','0','0','+45','+45','+90','+90']
windDirLabels = ['-90°','-45°','0°','+45°','+90°']
windDirLabels_noSymbols = ['n90','n45','0°','p45','p90']
windDirLabelsIC = ['90° contra','45° contra','0° (frontal)','45° ipsi','90° ipsi'] #for ipsi/contra plotting
speedLabels = ['0 cm/s', '50 cm/s', '100 cm/s', '200 cm/s']
degree_sign= u'\N{DEGREE SIGN}'

numDirs = 5 # hard-coded: 5 directions presented in these experiments
numVel = 4 # hard-coded: 4 velocites presented (0, 50, 100, 200 cm/s)

numFrames = 480
xMin_traces = -50
xMax_traces = 650
samplerate = 10000
framerate = 60 # Hz
lenVideo = 8 # in seconds
postWindRecovery = 2#number of seconds after wind turnd off in which antennae/neural repsonse still going back to baseline
framesPostWindOnset = 7 #delay from wind onset to antennal-neural response (for comparing wind- to non-wind-induced antennal motions)

baseAvgSt = 0 #in seconds, time during trace to start steady state wind average
baseAvgLen = 1 #in seconds, how long a region to average over
windAvgSt = 2 #in seconds, time during trace to start steady state wind average
windAvgLen = 2 #in seconds, how long a region to average over
activateAvgSt = 1 #activation average response start
activateAvgLen = 1 #activation average response length


activateStart = 1 #in seconds
activateTime = 5 # number of seconds opto activation
activateStop = activateTime+activateStart
activatePreWind = 1 # number of seconds to plot pre-wind (quantify antennal deflection with activation)
windStart = 2
windTime = 2
windStop = 4
stimBar_velTuningHeight = 1
preLightOn = 0.5
postLightOn = 1.8
lenOnsetLight = postLightOn - activateStart# 0.5
preLightOff = 5.5 #0.5 s before light off
postLightOff = 7 #after light off
lenOffsetLight = activateStop - preLightOff# 0.5

autocorr_stopWind = 235
autocorr_startWind = 125
autocorr_maxlag = 109#

odorWindOn = 2
odorWindTime = 2
odorWindOff = odorWindOn+odorWindTime
excludeOnOffsetInds = 5

peakWindow = int(framerate/2) #snippet of raw antenna angle to grab for each active movement ~+/-500ms

stimBar_height = 0.7 #height of bar indicating wind stimulus
stimBar_height_allDir = 5 #height of bar indicating wind stimulus
scaleAng_rawTraces = 10 #scale bar for raw antenna tracking
scaleAng_width = 3
scaleAng_xLocation = -20
shiftYTraces = 10
shiftYTracesCrossFly = 20
extraPltSpaceForRawVm = shiftYTraces*2

stimBar_Vm_height = 0.5 #height of rectangle for Vm traces (indicates wind stimulus)
scaleVm_rawTraces = 5 #scale bar height for Vm (mV)
scaleVm_xLocation = -samplerate/4
scaleVm_width = samplerate*0.010

r2_angShift = 0.1
regrRound = 2 #significant digits for regression reporting on figure

config_path = {}
config_path['frontal'] = baseDirectory+'/SuverEtAl2023_DLC/'+"mn_activation_frontal-Marie-2021-06-04/config.yaml"

fontFamily = 'sans-serif'
fontName = 'Arial'
fontSize_angPair = 10
fontSize_stimBar = 10
fontSize_angScale = 10
fontSize_axis = 10

windDirNames = ['-90°', '-45°', '0°', '+45°', '+90°']
windOdorNames = ['-90°','odor -90°', '-45°', 'odor -45°', '0°', 'odor 0°', '+45°','odor +45°', '+90°', 'odor +90°']
fsize_sub = 16
fsize_title = 20
fsize_velTuning_sub = 12
fsize_title_tuning = 16
fsize_raster = 10

# define line thickness for raw antennal traces (e.g. odor experiment)
traceWidAvg = 2
traceWidRaw = 1

headAxisMarkers = {}
angPairNames = {}
angPairs = {}
thirdSegAngPairs = {}
angleTextRotation = {}
angPairAverageSets = {}

namesIpsiContra = ['ipsi2', 'ipsi3', 'contra2', 'contra3']

rightSecondInd = 0
rightAristaInd = 1
leftSecondInd = 2
leftAristaInd = 3

headAxisMarkers["frontal"] = [(0, 2), (16, 18)] #points to define head axis
angPairNames["frontal"] = ['R_second_seg','R_third_seg', 'L_second_seg', 'L_third_seg']
angPairAverageSets["frontal"] = [np.array([0, 1]), np.array([3,4])]#[np.array([0, 1, 2]), np.array([4, 5, 6])]

#angPairs["frontal"] = [(3, 4),(5, 6),(7, 8), #right 2nd segment hairs (3, base-tip)
#                        (12, 13), #right arista tip-base
#                        (19, 20),(21, 22),(23, 24), #left 2nd segment hairs
#                        (28, 29)] #angles to compute relative to head axis
angPairs["frontal"] = [(6, 5),(8, 7),#(4, 3),(6, 5),(8, 7), #right 2nd segment hairs (base-tip)
                        (12, 13), #right arista tip-base
                        (22, 21),(24, 23),#(20, 19),(22, 21),(24, 23), #left 2nd segment hairs
                        (28, 29)] #angles to compute relative to head axis


angleTextRotation["frontal"] = 0


bodypartsOfInterest = {}
bodypartsOfInterest["frontal"] = [0,2,5,6,7,8,12,13,16,18,21,22,23,24,28,29]#[0,2,3,4,5,6,7,8,12,13,16,18,19,20,21,22,23,24,28,29]
bodypartsOfInterestNames = {}
bodypartsOfInterestNames["frontal"] = ['r_orbital_setae_dorsal',
'r_orbital_setae_anterior',
#'r_second_hair_1_base',
#'r_second_hair_1_tip',
'r_second_hair_2_base',
'r_second_hair_2_tip',
'r_second_hair_3_base',
'r_second_hair_3_tip',
'r_arista_tip',
'r_arista_base',
'l_orbital_setae_dorsal',
'l_orbital_setae_anterior',
#'l_second_hair_1_base',
#'l_second_hair_1_tip',
'l_second_hair_2_base',
'l_second_hair_2_tip',
'l_second_hair_3_base',
'l_second_hair_3_tip',
'l_arista_tip',
'l_arista_base']

bodypartsOfInterestWithFun = {}
bodypartsOfInterestWithFun["frontal"] = [0,2,3,4,5,6,7,8,12,13,15,16,18,19,20,21,22,23,24,28,29,31]
bodypartsOfInterestWithFunNames = {}
bodypartsOfInterestWithFunNames["frontal"] = ['r_orbital_setae_dorsal',
'r_orbital_setae_anterior',
'r_second_hair_1_base',
'r_second_hair_1_tip',
'r_second_hair_2_base',
'r_second_hair_2_tip',
'r_second_hair_3_base',
'r_second_hair_3_tip',
'r_arista_tip',
'r_arista_base',
'r_third_tip',
'l_orbital_setae_dorsal',
'l_orbital_setae_anterior',
'l_second_hair_1_base',
'l_second_hair_1_tip',
'l_second_hair_2_base',
'l_second_hair_2_tip',
'l_second_hair_3_base',
'l_second_hair_3_tip',
'l_arista_tip',
'l_arista_base',
'l_third_tip']


dotsize_bodypart = 3 #size of bodypart dot

#define colors and transparencies!

# for black background (presentations)
figColor = [0,0,0]
axisColor = [0.9,0.9,0.9]

# for white background (manuscript, Kathy presentations)
figColor = [1,1,1]
axisColor = [0.1,0.1,0.1]

peakMarker1 = np.array([246,193,65])/255
peakMarker2 = np.array([232,96,28])/255

windOnColor = 'blue' #color for movements at nset of wind
windOffColor = 'yellow' #color for movements at offset of wind

paleGray = np.array([221,221,221])/255
rasterColor = paleGray
gray = np.array([221,221,221])/255

lightBlue = np.array([166,206,227])/255
blue = np.array([31,120,180])/255
paleGreen = np.array([178,223,138])/255
green = np.array([51,160,44])/255
salmon = np.array([251,154,153])/255
red = np.array([227,26,28])/255
creamsicle = np.array([253,191,111])/255
orange = np.array([255,127,0])/255
lightUrple = np.array([202,178,214])/255
purple = np.array([106,61,154])/255
lightYellow = np.array([255,255,153])/255
brown = np.array([177,89,40])/255

singleFlyRasterColors = [lightBlue, blue, paleGreen,green,salmon,red,
     creamsicle,orange,lightUrple,purple,lightYellow,brown]

blue = np.array([68,119,170])/255
cyan = np.array([102,204,238])/255
purple =np.array([170,51,119])/255
red = np.array([238,102,119])/255
rasterColorCompare = [paleGray,blue, cyan,purple, red]

cyan = np.array([51,187,238])/255
teal = np.array([0,153,136])/255
orange = np.array([238,119,51])/255
red = np.array([204,51,17])/255
magenta = np.array([238,51,119])/255

green32 = np.array([17,119,51])/255
purple32 = np.array([170,68,153])/255

colors_antAngs = [cyan, teal, orange, magenta]

cyanD = np.array([41,177,228])/255
tealD = np.array([0,143,126])/255
orangeD = np.array([228,109,41])/255
magentaD = np.array([228,41,119])/255
colors_antAngsDark = [cyanD, tealD, orangeD, magentaD]

colors_directions = [purple, blue, teal, orange, magenta]


seq1 = np.array([251,154,41])/255 #0 cm/s (light orange)
seq2 = np.array([236,112,20])/255 #50 cm/s (dark orange)
seq3 = np.array([204,76,2])/255 #100 cm/s (orangey-red)
seq4 = np.array([153,52,4])/255 #200 cm/s (brown-red)
colors_velocity_right = [seq1, seq2, seq3, seq4]

seq1_l = np.array([181,221,216])/255 #0 cm/s (greenish blue)
seq2_l = np.array([129,196,231])/255 #50 cm/s (blue)
seq3_l = np.array([47,152,210])/255 #100 cm/s (light purple)
seq4_l = np.array([144,99,136])/255 #200 cm/s (dark purple)
colors_velocity_left = [seq1_l, seq2_l, seq3_l, seq4_l]

light_gray = np.array([228,224,220])/255 #0 cm/s (greenish blue)
medLight_gray = np.array([186,176,164])/255 #50 cm/s (blue)
med_gray = np.array([143,127,109])/255 #100 cm/s (light purple)
dark_gray = np.array([87, 78, 67])/255 #200 cm/s (dark purple)
colors_velocity_RminL = [light_gray, medLight_gray, med_gray, dark_gray]
flight_color = med_gray

markerColor_diffIndv = med_gray
markerColor_diffAvg = medLight_gray

markerTuningIndv = 10
markerTuningAvg = 16
yaxis_min_tuning = -20
yaxis_max_tuning = 20

rasterMkrSizeAllDirs = 3


markerColor_74C10 = red
markerColor_18D07 =purple
markerColor_CS = med_gray
markerColor_91F02 = blue

color_rawVmTrace = 'grey'
color_windStim = 'grey'
color_activateColor = np.array([255,40,0])/255 #ferrari red ;)
color_inactivateColor = np.array([94,255,0])/255#np.array([74,150,208])/255 #blue


color_inactivationNoLight = med_gray
color_inactivationLight = color_inactivateColor

transparencyAntTrace = 0.5
transparencyTuningPoints = 0.7
transparencyActiveTrace = 0.2
transparencyAllActiveTraces = 0.05
transparencyAllSegmentTraces = 0.2
transparencyHist = 0.7
transparencySingleFlyTrace = 0.3
transparencyCrossFlyTrace = 0.9
transparencyPatch = 0.2

violinColor = 'grey'

color_rightAnt = teal
color_leftAnt = orange

#define a sequential (discrete rainbow scheme) for increasing wind velocities
green = np.array([78,178,101])/255
mint = np.array([202,224,171])/255
light_orange = np.array([246,193,65])/255
dark_orange = np.array([232,96,28])/255

green = np.array([144,201,135])/255
mint = np.array([247,240,86])/255
light_orange = np.array([241,147,45])/255
dark_orange = np.array([220,5,12])/255
colors_vel = [green, mint, light_orange, dark_orange]

color_quant_wholeTrace = 'grey'
color_quant_wind = 'red'
color_quant_notWind = 'cyan'
