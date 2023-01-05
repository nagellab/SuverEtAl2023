# Plotting functions for Figures in Suver et al. 2023


import importlib as il
import plotting as pt  #c ontains most of the analysis and plotting functions
import loadNotes_mnOptoVelocityTuning as ln

# FIGURE 1

#Figure 1B & supplement - example frame with tracking points, head axis
# Plot image with tracking points (unfortunately the save feature is not working, you'll have to save manually):
# image with no points
pt.plot_head_markers_axis('2020_12_14_E3',trialNum=41,frameNum=1,lines=-1,headaxis=0,savefig=1)
#first plot all points of interest with relative antennal angles, including head axis
pt.plot_head_markers_axis('2020_12_14_E3',trialNum=41,frameNum=1,lines=1,headaxis=1,savefig=1)
#then plot just antenna angles
pt.plot_head_markers_axis('2020_12_14_E3',trialNum=41,frameNum=1,lines=2,headaxis=0,savefig=1)
#then plot all head points tracked for supplemental figure
pt.plot_head_markers_axis('2020_12_14_E3',trialNum=41,frameNum=1,lines=0,headaxis=0,savefig=1)

# 1C - mostly passive + active movement example
pt.getAntennaTracesAvgs_singleExpt('2021_02_26_E1',TEST=42)  #frontal wind, mostly passive, 200cm/s
pt.getAntennaTracesAvgs_singleExpt('2021_03_10_E5',TEST=42) #nice active on top of passive example, 200 cm/s

# 1D, 1E - second and third segment traces across directions and velocities
pt.plot_cross_fly_traces(expt='CS_activate',cameraView='frontal',singleDir=-1, avgOnly = 0, allowFlight=0,plotInactivate=0,savefig=1)

# 1F (slow adaptation):
pt.plot_single_expt_traces(expt='2020_12_14_E3',importAnew=1,plotInactivate=0,savefig=1)

# 1G: Plot second segment deflection across the different speeds
pt.plot_second_displacement_velocity(direction=2,savefig=1) #plots for just frontal (0 deg) direction

# Supplemental 1G: Plot second segment deflection across the different speeds
pt.plot_second_displacement_velocity(direction=-1,savefig=1) #plots across all directions (pools them)

#1J - raster
pt.plot_active_movement_raster_cross_fly(expt='CS_activate',cameraView='frontal', plotInactivate=0,importAnew = 0,savefig = 1,secondSeg=1)

#1K - active movements during wind
pt.plot_movement_count_during_wind(expt='CS_activate',cameraView='frontal',importAnew = 0,savefig = 1,secondSeg=1)

#1M - Cross-correlation
pt.plot_xcorr_no_wind(cameraView='frontal',import_anew = 0,savefig=0)

#Figure 1N, 1O - odor and flight raster and violin plots (prints means, quartiles for violin)
pt.plot_active_movement_raster_cross_fly_odor()

# Supplemental Figure 2: light activation response for second and third segments
pt.plot_lightResponse('CS_activate',savefig=1,importAnew=0)
pt.plot_lightResponse('18D07_activate',savefig=1,importAnew=0)
pt.plot_lightResponse('91F02_activate',savefig=1,importAnew=0)
pt.plot_lightResponse('74C10_activate',savefig=1,importAnew=0)

#Figure 3F,I plot inactivation psth for the three conditions, overlaid
pt.plot_psth_inactivation(savefig=1)

#Figure 3A,B,C right column:  - inactivation no light vs. light quantification of active movements (classified using 2nd antennal segment)
pt.plot_inactivation_paired_active_movements(expt='18D07_inactivate', cameraView='frontal',importAnew=0,savefig=1,secondSeg=1)
pt.plot_inactivation_paired_active_movements(expt='emptyGAL4_inactivate', cameraView='frontal',importAnew=0,savefig=1,secondSeg=1)
pt.plot_inactivation_paired_active_movements(expt='91F02_inactivate', cameraView='frontal',importAnew=0,savefig=1,secondSeg=1)

#Supplemental Figure 3A,B,C: inactivation traces
pt.plot_lightResponse('18D07_inactivate',savefig=1,importAnew=0)
pt.plot_lightResponse('91F02_inactivate',savefig=1,importAnew=0)
pt.plot_lightResponse('emptyGAL4_inactivate',savefig=1,importAnew=0)

# Figure 3 Supplement (D): plot comparison of inactivation traces
pt.plot_light_response_quant_inactivation(savefig=1)

#Figure 4 activation:
# These will plot velocity and direction tuning data for the 4 genotypes
# deflections will be third-second segment deflections relative to the midline
pt.plot_wind_vel_tuning('CS_activate',savefig=1,importAnew=0)
pt.plot_wind_vel_tuning('18D07_activate',savefig=1,importAnew=0)
pt.plot_wind_vel_tuning('91F02_activate',savefig=1,importAnew=0)
pt.plot_wind_vel_tuning('74C10_activate',savefig=1,importAnew=0)
pt.plot_gain_tuning_quant(savefig=1,importAnew=0) #plots the -45 - +45 average response difference across all 4 speeds, all genotypes
