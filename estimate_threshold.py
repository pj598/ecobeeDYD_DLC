#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:54:40 2021

@author: pamelawildstein
"""

import pandas as pd 
import numpy as np 
from datetime import timedelta
import matplotlib.pyplot as plt 
import seaborn as sns
from beepy import beep 

###double checking stuff#######
##################VARIABLES###################################################
drParticipants = '' #list of DYD IDs you want the thresholds for 
startEndTimes = '/' #start and end times of events
originalDFLocation = '' #full profile for thermostat from DYD
identifierYear = 2019
timeZone = 'America/Los_Angeles'
finalDFPathway ='' #final pathway for where you want it to go
folds = 10
##############################################################################
pd.options.mode.chained_assignment = None #CAUTION GO BACK AND CHECK THE WARNINGS LATER
#demand response timeframes
demandResponseDates = ['7/24/19','8/13/19-1','8/13/19-2','8/14/19','8/15/19',
                              '8/21/19-1','8/21/19-2','8/26/19','8/27/19','8/28/19',
                              '9/3/19','9/4/19','9/5/19','9/8/19','9/12/19','9/13/19',
                              '9/24/19-1','9/24/19-2','9/25/19','10/16/19','10/21/19',
                              '10/22/19']

timesDF = pd.read_csv(str(startEndTimes))
timesDF['START'] = pd.to_datetime(timesDF['START'],errors='raise',utc=True).dt.tz_convert(str(timeZone))
timesDF['END'] = pd.to_datetime(timesDF['END'],errors='raise',utc=True).dt.tz_convert(str(timeZone))



#get DR participatins 
columns1 = ['Identifier','7/24/19','8/13/19-1','8/13/19-2','8/14/19','8/15/19',
           '8/21/19-1','8/21/19-2','8/26/19','8/27/19','8/28/19',
            '9/3/19','9/4/19','9/5/19','9/8/19','9/12/19','9/13/19',
            '9/24/19-1','9/24/19-2','9/25/19','10/16/19','10/21/19',
            '10/22/19','DR']

drParticipants = pd.read_csv(str(drParticipants),usecols=columns1)
drParticipantsTrue = drParticipants[drParticipants['DR'] == True]
drParticipantsList = drParticipantsTrue['Identifier'].unique().tolist()



columns2 = ['DATE','HvacMode','CalendarEvent','Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat',
            'fan','compCool1','compCool2','HourlyDryBulbTemperature']
cols = ['Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat',
            'fan','compCool1','compCool2','HourlyDryBulbTemperature']
colsFix = ['Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat']

dropCols = ['TemperatureExpectedCool','fan','compCool1','compCool2','HourlyDryBulbTemperature'] #if these columns are missing values, drop them

#stuff 
counter = 0
length = len(drParticipantsList)
finalDF = pd.DataFrame()
moreOutliersList = []
finalDF = pd.DataFrame()

for identifier in drParticipantsList:
    #general prep
    print("CURRENT IDENTIFIER:" + str(identifier))
    newDF = pd.DataFrame({'identifier':[str(identifier)]})
    originalDF = pd.read_csv(str(originalDFLocation) + str(identifier) + '_' + str(identifierYear) + '.csv', usecols = columns2).drop_duplicates(keep='first')
    originalDF['DATE'] = pd.to_datetime(originalDF['DATE'],errors='coerce',utc=True).dt.tz_convert(str(timeZone))
    originalDF= originalDF.set_index('DATE').sort_index()
  
    originalDF[cols] = originalDF[cols].apply(pd.to_numeric, errors='coerce')
    originalDF[colsFix] = originalDF[colsFix]/10
    originalDF= originalDF.dropna(subset=dropCols)
    
    originalDF_edited = originalDF
    originalDF_edited['outdoorTemp_lag_1'] = originalDF_edited['HourlyDryBulbTemperature'].shift(periods=1)
    originalDF_edited['outdoorTemp_lag_2'] = originalDF_edited['HourlyDryBulbTemperature'].shift(periods=2)
    originalDF_edited['Temperature_Ctrl_lag_1'] = originalDF_edited['Temperature_Ctrl'].shift(periods=1)
    originalDF_edited['Temperature_Ctrl_lag_2'] = originalDF_edited['Temperature_Ctrl'].shift(periods=2)
    
    originalDF_edited['month'] = originalDF_edited.index.month
    originalDF_edited['hour'] = originalDF_edited.index.hour
    
    originalDF_edited['fan_on_off'] = np.where(originalDF_edited['fan'] > 0, 1, 0)
    originalDF_edited['fan_on_off_lag_1'] = originalDF_edited['fan_on_off'].shift(1)
    
    originalDF_edited['compCool'] = originalDF_edited['compCool1'] + originalDF_edited['compCool2']
    originalDF_edited['compCool_on_off'] = np.where(originalDF_edited['compCool'] > 0, 1, 0)
    originalDF_edited['both_on'] = (originalDF_edited['fan_on_off'] == 1) & (originalDF_edited['compCool_on_off'] == 1) 

######originalDF_edited is the OG DF, we will not be changing it again#################
    #give the trainer DF the on/off and on/off lagged column
    df = originalDF_edited.sort_index()


    df = df.loc[df.index.month.isin(list(range(6,11)))]
    df['auto_cool'] = 0
    


    df.loc[(df['HvacMode']=='auto') | (df['HvacMode'] == 'cool'), 'auto_cool'] = 1
    df.loc[(df['CalendarEvent'].isin(['SmartRecovery','smartAway','smartHome','HKhold'])), 'auto_cool'] = 0  #if the calendar event is smart recovery, we aren't considering it

    
    intervals = 12

    for t in list (range(intervals)):
        df['auto_cool'+str(t)] = df['auto_cool'].shift(t)
    

    df = df.dropna(subset = ['auto_cool' + str(t) for t in list (range(intervals))]) #dropping the nans to avoid problems before they occur

    df['AC_streak'] = df[['auto_cool' + str(t) for t in list (range(intervals))]].sum(axis=1) #THIS WILL NOT WORK UNLESS AXIS IS SET TO 1
    df['continuous_stretch'] = 0
    df['continuous_stretch'] = np.where(df['AC_streak'] >= (intervals-1),1,0)
 
    streakRows = df.loc[df['continuous_stretch'] == 1]

    
    streakRows['time'] = streakRows.index
    streakRows['tdiff'] = streakRows['time']-streakRows['time'].shift()
    discontRows = streakRows.loc[streakRows['tdiff'] > pd.Timedelta(5,'min')]
    #find the rows where the difference between it and the row before it are more than 5 min
                                                                 #because that is where the blocks start 
    allDeltaT = pd.DataFrame()  #creating a list 

    for idx in range(discontRows.shape[0]-1):  
        

        streak = streakRows.loc[discontRows.iloc[idx].name:discontRows.iloc[idx+1].name] #streak = the original DF but filtered for the blocks that we want
        print(streak)
        hvac_ch = streak[(streak['fan_on_off'] - streak['fan_on_off_lag_1']) > 0 ]  #hvac change -- is the fan turning on? 
        hvac_ch = hvac_ch[hvac_ch['auto_cool'] == 1]
        hvac_ch = hvac_ch[hvac_ch['both_on'] == True]
        deltaT = hvac_ch['Temperature_Ctrl'] - hvac_ch['TemperatureExpectedCool'].values  #deltaT = indoor temp - setpoint
        deltaTDf = pd.DataFrame({'deltaT':deltaT})
        allDeltaT = pd.concat([allDeltaT,deltaTDf])
        allDeltaT.append(deltaT)#add to the list 
    

    #dataframe w/ results 
    newDF = pd.DataFrame({'identifier':[str(identifier)]})
    try:
        newDF['mean'] = allDeltaT['deltaT'].mean()
        print(newDF)
        newDF['mode'] = allDeltaT['deltaT'].mode()
        newDF['median'] = allDeltaT['deltaT'].median()
    except KeyError:
        newDF['mean'] ='not enough rows'
        newDF['mode'] = 'not enough rows'
        newDF['median'] = 'not enough rows'
        
    finalDF = pd.concat([finalDF, newDF])
    finalDF.to_csv(finalDFPathway + 'threshold_estimation.csv')
    

    
    allDeltaT = allDeltaT.reset_index()
    #graph

        
        