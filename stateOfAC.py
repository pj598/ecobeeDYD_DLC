#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 01:10:23 2021

@author: pamelawildstein
"""

"""
Last updated: 12/16/2021

This file has the indoor temperature regression and the control logic of the thermostat.
It will only tell you if the AC is on or off for the five minute intervals for no DLC participation (expected) and no override (perfect).
    Pardon the change in wording, we changed terminology long after I wrote and used this code. 

It will not tell you the AC unit capacity (via restock data) or the electricity usage (via Goodmman catalog specs). 
It will not tell you the ELCC of the program
    Those are other .py files that use these results in the github 


What did we do before this?
1. Found weather stations for each home through NOAA's Local Climatological Data too. 
Then used the weather station data to get the HourlyDryBulbTemperature (the outdoor temperature)
(https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data)
2. Determined the thermostats in our sample (aka the list of thermostats in drParticipantsPathway)
3. Estimated the thresholds for each thermostat (data in threshold pathway)
4. Got the start and end times of events. 
5. Got the minimum outdoor temperature each thermostat experienced during the summer (hotTemps file)


"""

import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from datetime import timedelta


def main():
    originalDFLocation, startEndTimes, identifierYear, timeZone, finalDFPathway, thresholdPathway, hotTemps, drParticipantsPathway = definePathways()
    
    timesDF, thresholdsDF, hotTempsDF, drParticipants = importDataFrames(startEndTimes, 
                                                                         thresholdPathway, timeZone, hotTemps,
                                                                         drParticipantsPathway)
    identifierList, confusionDF, finalTempStats, outliersList = initializeStuff(drParticipants)
    
    counter = 1
    
    for identifier in identifierList:
        confusionNew, tempNew, outliersList = modelHomes(timesDF, identifier, hotTempsDF, timeZone, identifierYear, finalDFPathway, thresholdsDF,
               startEndTimes, originalDFLocation, drParticipants, outliersList)
        
        confusionDF = pd.concat([confusionDF, confusionNew])
        confusionDF.to_csv(finalDFPathway + 'confusionMatrix.csv')
        
        finalTempStats = pd.concat([finalTempStats, tempNew])
        finalTempStats.to_csv(finalDFPathway + 'regressionAccuracy.csv')
        print(str(counter) + '/' + str(len(identifierList)))
        counter +=1


##################VARIABLES###################################################
def definePathways():
    originalDFLocation = '' #original thermostat profile you get from ecobee dyd 
    startEndTimes = ''  #start and end times of DLC events
    identifierYear = 2019  #the year 
    timeZone = 'America/Los_Angeles'   #time zone 
    finalDFPathway ='' #where you want the dataframe with results to go
    thresholdPathway = '' #df with estimamted thresholds 
    hotTemps = '' #df with minimmum outdoor temp thermostat experienced during the summer 
    drParticipantsPathway = '' #a df that has all the events as columns, the thermostat IDs as rows, and True/False if they participated in the event 
    
    return originalDFLocation, startEndTimes, identifierYear, timeZone, finalDFPathway,thresholdPathway, hotTemps, drParticipantsPathway
##############################################################################

def importDataFrames(startEndTimes, thresholdPathway, timeZone, hotTemps, drParticipantsPathway):
    timesDF = pd.read_csv(str(startEndTimes))
    timesDF['START'] = pd.to_datetime(timesDF['START'],errors='raise',utc=True).dt.tz_convert(str(timeZone))
    timesDF['END'] = pd.to_datetime(timesDF['END'],errors='raise',utc=True).dt.tz_convert(str(timeZone))
    
    thresholdsDF = pd.read_csv(thresholdPathway,usecols=['identifier','new_median','HVAC_on']).set_index('identifier')
    thresholdsDF = thresholdsDF[thresholdsDF['HVAC_on']==True]
    thresholdsDF['new_median'] = pd.to_numeric(thresholdsDF['new_median'])

    
    hotTempsDF = pd.read_csv(hotTemps,index_col='Identifier')
    
    drParticipants = pd.read_csv(drParticipantsPathway).set_index('Identifier')
    
    return timesDF, thresholdsDF, hotTempsDF, drParticipants
def initializeStuff(drParticipants):
    identifierList = drParticipants[drParticipants.DR == True].index.tolist()
    print(len(identifierList))
    confusionDF = pd.DataFrame()
    finalTempStats = pd.DataFrame()
    outliersList = np.array([])
    
    return identifierList, confusionDF, finalTempStats, outliersList

def modelHomes(timesDF, identifier, hotTempsDF, timeZone, identifierYear, finalDFPathway, thresholdsDF,
               startEndTimes, originalDFLocation, drParticipants, outliersList):
    originalDF_edited = importAndCleanDF(identifier, originalDFLocation, timeZone, identifierYear)
    streakRows, discontRows, temp = setUpDF(originalDF_edited, hotTempsDF,identifier)
    streakFull = testBlocks(streakRows, discontRows, identifier)
    finalDFHome, tempNew, regTemp, thresholdHome  = trainModel(streakFull, identifier, streakRows, finalDFPathway, discontRows, thresholdsDF, timeZone)
    confusionNew = testModel(finalDFHome, identifier, thresholdHome, temp)
    outliersList = predictModel(startEndTimes, finalDFPathway, identifier, timeZone, originalDFLocation,identifierYear, timesDF, drParticipants, regTemp, thresholdHome, outliersList)
    
    return confusionNew, tempNew, outliersList

def importAndCleanDF(identifier, originalDFLocation, timeZone, identifierYear):
    print("CURRENT IDENTIFIER:" + str(identifier))
    
    cols = ['Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat',
            'fan','compCool1','compCool2','HourlyDryBulbTemperature']
    
    columns2 = ['DATE','HvacMode','CalendarEvent','Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat',
            'fan','compCool1','compCool2','HourlyDryBulbTemperature']
    
    colsFix = ['Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat']

    dropCols = ['TemperatureExpectedCool','fan','compCool1','compCool2','HourlyDryBulbTemperature']
    
    newDF = pd.DataFrame({'identifier':[str(identifier)]})
    originalDF = pd.read_csv(str(originalDFLocation) + str(identifier) + '_' + str(identifierYear) + '.csv', usecols = columns2).drop_duplicates(keep='first')
    originalDF['DATE'] = pd.to_datetime(originalDF['DATE'],errors='coerce',utc=True).dt.tz_convert(str(timeZone))
    originalDF= originalDF.set_index('DATE').sort_index()
  
    originalDF[cols] = originalDF[cols].apply(pd.to_numeric, errors='coerce')
    originalDF[colsFix] = originalDF[colsFix]/10
    originalDF= originalDF.dropna(subset=dropCols)
    
    originalDF_edited = originalDF
    originalDF_edited['compCool'] = originalDF_edited['compCool1'] + originalDF_edited['compCool2']

    originalDF_edited['Temperature_Ctrl_lag_1'] = originalDF_edited['Temperature_Ctrl'].shift(periods=1)
    originalDF_edited['Temperature_Ctrl_lag_2'] = originalDF_edited['Temperature_Ctrl'].shift(periods=2)
    
    originalDF_edited['HourlyDryBulbTemperature_lag_1'] = originalDF_edited['HourlyDryBulbTemperature'].shift(periods=1)
    
    originalDF_edited['month'] = originalDF_edited.index.month
    originalDF_edited['hour'] = originalDF_edited.index.hour
    
    
    originalDF_edited['fan_on_off'] = np.where(originalDF_edited['fan'] > 0, 1, 0)
    
    originalDF_edited['compCool'] = originalDF_edited['compCool1'] + originalDF_edited['compCool2']
    originalDF_edited['compCool_on_off'] = np.where(originalDF_edited['compCool'] > 0, 1, 0)
    originalDF_edited['compCool_on_off_lag_1'] = originalDF_edited['compCool_on_off'].shift(1)
    originalDF_edited['both_on'] = (originalDF_edited['fan_on_off'] == 1) & (originalDF_edited['compCool_on_off'] == 1) 
    
    return originalDF_edited

def setUpDF(originalDF_edited, hotTempsDF,identifier):
    df = originalDF_edited.sort_index()
    df = df.loc[df.index.month.isin(list(range(6,11)))]
    
    temp = hotTempsDF.loc[str(identifier),'min']
    
    df.loc[(df['HvacMode']=='auto') | (df['HvacMode'] == 'cool'), 'auto_cool'] = 1
    df.loc[(df['CalendarEvent'].isin(['SmartRecovery','smartAway','smartHome','HKhold'])), 'auto_cool'] = 0  

    intervals = 13

    for t in list (range(intervals)):
        df['auto_cool'+str(t)] = df['auto_cool'].shift(t)
    

    df = df.dropna(subset = ['auto_cool' + str(t) for t in list (range(intervals))]) 
    df['AC_streak'] = df[['auto_cool' + str(t) for t in list (range(intervals))]].sum(axis=1) 
    df.loc[:,'continuous_stretch'] = 0
    df['continuous_stretch'] = np.where(df['AC_streak'] >= (intervals-1),1,0)
 
    streakRows = df.loc[df['continuous_stretch'] == 1]
    
    streakRows = streakRows.loc[streakRows['auto_cool'] == 1]
    streakRows = streakRows.loc[streakRows['HourlyDryBulbTemperature'] >= temp]
    streakRows = streakRows.loc[streakRows['fan_on_off'] == streakRows['compCool_on_off']]
    streakRows['time'] = streakRows.index
    streakRows.loc[:,'tdiff'] = streakRows['time']-streakRows['time'].shift()
    discontRows = streakRows.loc[streakRows['tdiff'] > pd.Timedelta(5,'min')]
    
    
    return streakRows, discontRows, temp

def testBlocks(streakRows, discontRows, identifier):
    streakFull = pd.DataFrame()
    
    for idx in range(discontRows.shape[0]-1):  
        try:

            streak = streakRows.loc[discontRows.iloc[idx].name:discontRows.iloc[idx+1].name] 
            streak = streak.drop(streak.tail(1).index)
            newStreak = streak.iloc[1:]
            streakFull = pd.concat([streakFull,newStreak])
            
        except KeyError and IndexError:
            continue
        
    return streakFull

def trainModel(streakFull, identifier, streakRows, finalDFPathway, discontRows, thresholdsDF, timeZone):
    trainerDF = streakFull

    tempNew = pd.DataFrame({'identifier':[str(identifier)]})
    thresholdHome = thresholdsDF.loc[str(identifier),'new_median']
    
    try:
        trainerDF = trainerDF.dropna(subset=['Temperature_Ctrl_lag_1',
                                      'compCool_on_off_lag_1'])
    
    

        tempTrainX = trainerDF[['Temperature_Ctrl_lag_1','Temperature_Ctrl_lag_2','HourlyDryBulbTemperature','HourlyDryBulbTemperature_lag_1',
                                'compCool_on_off_lag_1']]
        tempTrainY = trainerDF['Temperature_Ctrl']
        regTemp = LinearRegression().fit(tempTrainX,tempTrainY)
        tempNew['RMSE'] = np.absolute(cross_val_score(regTemp,tempTrainX,tempTrainY, cv=10,scoring='neg_root_mean_squared_error')).mean()
        tempNew['R2'] = cross_val_score(regTemp,tempTrainX,tempTrainY, cv=10,scoring='r2').mean()
        tempNew = tempNew.set_index('identifier')

        
    except KeyError:
        print("there is an issue")
    
    finalDFHome = pd.DataFrame()
    for idx in range(discontRows.shape[0]-1):  
        try:
            streak = streakRows.loc[discontRows.iloc[idx].name:discontRows.iloc[idx+1].name]
            streak = streak.drop(streak.tail(1).index)
            streak = streak.iloc[1:]
            fan_index = streak.index.tolist()
            Temperature_Ctrl_lag_1 = [streak['Temperature_Ctrl_lag_1'][0]]
            Temperature_Ctrl_lag_2 = [streak['Temperature_Ctrl_lag_2'][0]]
            HourlyDryBulbTemperature_lag_0 = streak['HourlyDryBulbTemperature'].tolist()
            HourlyDryBulbTemperature_lag_1 = streak['HourlyDryBulbTemperature_lag_1'].tolist()

            
            compCool_result_expected_lag_1 = [streak['compCool_on_off_lag_1'][0]]
            TemperatureExpectedCool = streak['TemperatureExpectedCool'].tolist()
            compCool_on_off_predict = []
            TemperatureCtrl_predict = []
            
            i = 0
            while i < len(fan_index):
                temperatureCtrl = regTemp.predict([[Temperature_Ctrl_lag_1[i],Temperature_Ctrl_lag_2[i],HourlyDryBulbTemperature_lag_0[i],
                                                  HourlyDryBulbTemperature_lag_1[i],compCool_result_expected_lag_1[i]]])
                                       
                TemperatureCtrl_predict = TemperatureCtrl_predict + np.round(temperatureCtrl,2).tolist()
                
                difference = temperatureCtrl - TemperatureExpectedCool[i]
                
                #turn on 
                if compCool_result_expected_lag_1[i] == 0 and difference >= thresholdHome : 
                    compCool_on_off_predict_new = 1
                
                #stay on 
                elif compCool_result_expected_lag_1[i] ==  1 and difference > 0:
                    compCool_on_off_predict_new = 1
                
                #turn off
                elif  compCool_result_expected_lag_1[i] == 1 and difference <= 0:
                    compCool_on_off_predict_new = 0
                
                #stay off 
                else:
                    compCool_on_off_predict_new = 0            
                
                compCool_on_off_predict = compCool_on_off_predict + [compCool_on_off_predict_new]
    
                
                if i == len(fan_index)-1:
                    newDF = pd.DataFrame({'DATE':fan_index,'Temperature_Ctrl_lag_1_new': Temperature_Ctrl_lag_1,'Temperature_Ctrl_lag_2_new': Temperature_Ctrl_lag_2,
                        'compCool_result_expected_lag_1':compCool_result_expected_lag_1,'compCool_on_off_predict':compCool_on_off_predict,'Temperature_Ctrl_predict':TemperatureCtrl_predict})
                    newDF['DATE'] = pd.to_datetime(newDF['DATE'],errors='coerce',utc=True).dt.tz_convert(str(timeZone))
                    newDF = newDF.set_index('DATE').sort_index()
                    
                    finalResultsDF = streak.merge(newDF,how='right',left_index=True,right_index=True)
                    finalResultsDF['compCool_correct'] = finalResultsDF['compCool_on_off'] - finalResultsDF['compCool_on_off_predict']

                    finalDFHome = pd.concat([finalDFHome,finalResultsDF])
                    finalDFHome.to_csv(finalDFPathway + 'homes/' + str(identifier) + ".csv")
                    
                    i += 1
                    
    
                else:
                    compCool_result_expected_lag_1 = compCool_result_expected_lag_1 + [compCool_on_off_predict_new]
                    Temperature_Ctrl_lag_2 = Temperature_Ctrl_lag_2 + [Temperature_Ctrl_lag_1[i]]
                    Temperature_Ctrl_lag_1 = Temperature_Ctrl_lag_1 + np.round(temperatureCtrl,2).tolist()

                    i += 1

        except KeyError and IndexError: 
            continue
        
    return finalDFHome, tempNew, regTemp, thresholdHome 

def testModel(finalDFHome, identifier, thresholdHome, temp):
    confusionComp = metrics.confusion_matrix(finalDFHome['compCool_on_off'],finalDFHome['compCool_on_off_predict'])
    confusionNew = pd.DataFrame({'identifier':[str(identifier)]})
    confusionNew = confusionNew .set_index('identifier')
    confusionNew['threshold'] = str(thresholdHome)
 
    try:
        confusionNew['compCool_true_positive'] = confusionComp[1,1]
        confusionNew['compCool_true_negative'] = confusionComp[0,0]
        confusionNew['compCool_false_postivie'] = confusionComp[0,1]
        confusionNew['compCool_false_negative'] = confusionComp[1,0]
    except IndexError:
        pass
    try:
        confusionNew['compCool_accuracy_score'] = metrics.accuracy_score(finalDFHome['compCool_on_off'],finalDFHome['compCool_on_off_predict']) 
        confusionNew['compCool_sensitivity'] =  metrics.recall_score(finalDFHome['compCool_on_off'],finalDFHome['compCool_on_off_predict']) 
        confusionNew['compCool_precision'] = metrics.precision_score(finalDFHome['compCool_on_off'],finalDFHome['compCool_on_off_predict']) 
        confusionNew['compCool_RMSE_result'] = metrics.mean_squared_error(finalDFHome['compCool_on_off'],finalDFHome['compCool_on_off_predict'],squared=False)
        confusionNew['temp_floor'] = temp
    except KeyError:
        pass

    return confusionNew

def predictModel(startEndTimes, finalDFPathway, identifier, timeZone, originalDFLocation,identifierYear, timesDF, drParticipants, regTemp, thresholdHome, outliersList):
    demandResponseDates = ['7/24/19','8/13/19-1','8/13/19-2','8/14/19','8/15/19',
                              '8/21/19-1','8/21/19-2','8/26/19','8/27/19','8/28/19',
                              '9/3/19','9/4/19','9/5/19','9/8/19','9/12/19','9/13/19',
                              '9/24/19-1','9/24/19-2','9/25/19','10/16/19','10/21/19',
                              '10/22/19']
    
    cols = ['Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat',
            'fan','compCool1','compCool2','HourlyDryBulbTemperature']
    
    columns2 = ['DATE','HvacMode','CalendarEvent','Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat',
            'fan','compCool1','compCool2','HourlyDryBulbTemperature']
    
    colsFix = ['Temperature_Ctrl','TemperatureExpectedCool','TemperatureExpectedHeat']

    dropCols = ['TemperatureExpectedCool','fan','compCool1','compCool2','HourlyDryBulbTemperature']
    
    originalDF = pd.read_csv(str(originalDFLocation) + str(identifier) + '_' + str(identifierYear) + '.csv', usecols = columns2).drop_duplicates(keep='first')
    originalDF['DATE'] = pd.to_datetime(originalDF['DATE'],errors='coerce',utc=True).dt.tz_convert(str(timeZone))
    originalDF= originalDF.set_index('DATE').sort_index()

    
    originalDF[cols] = originalDF[cols].apply(pd.to_numeric, errors='coerce')
    originalDF[colsFix] = originalDF[colsFix]/10
    
    originalDF['compCool'] = originalDF['compCool1'] + originalDF['compCool2']
    originalDF['compCool_on_off'] = np.where(originalDF['compCool'] > 0, 1, 0)
    originalDF['compCool_on_off_lag_1'] = originalDF['compCool_on_off'].shift(1)
    
    originalDF['Temperature_Ctrl_lag_1'] = originalDF['Temperature_Ctrl'].shift(periods=1)
    originalDF['Temperature_Ctrl_lag_2'] = originalDF['Temperature_Ctrl'].shift(periods=2)
    
    
    originalDF['HourlyDryBulbTemperature_lag_1'] = originalDF['HourlyDryBulbTemperature'].shift(periods=1)
    originalDF['DR_event'] = False
    originalDF= originalDF.dropna(subset=dropCols)
    

    
    #thou shalt not mess with the originalDF copy
    testDF = pd.DataFrame()
    drParticipantsTrue = drParticipants[drParticipants['DR'] == True]


    for date, timeStart, timeEnd in zip(demandResponseDates,timesDF['START'],timesDF['END']) :
        if drParticipantsTrue.loc[str(identifier), str(date)] == True:
            try:

                timeEnd5minutes = timeEnd - timedelta(minutes=5)
                newDF = originalDF.loc[timeStart:timeEnd5minutes]
                
                #times for later
                timeStart10minutes = timeStart - timedelta(minutes=10)
                
                #add correct values into the columns  FIGURE OUT TO DO WITH THOSE LAST FIVE MINUTES
                newDF.loc[timeStart:timeEnd5minutes,'perfectResponse'] = originalDF.loc[timeStart,'TemperatureExpectedCool']
                newDF.loc[timeStart:timeEnd5minutes,'expectedSetPoint'] = originalDF.loc[timeStart10minutes,'TemperatureExpectedCool']
                testDF = pd.concat([testDF,newDF])
                

            except KeyError: 
                print('try/except')
                pass
        else:
            pass 
        
    
    fullDF = originalDF.drop(testDF.index)

    #deal with problem dates (i.e. dates where events overlapped because they were called at different places at about the same time) 
    #but ecobee doesn't tell me who is in which, so I have to commbine them into one
    try:
        if testDF.loc['2019-09-25 18:55:00-07:00','TemperatureExpectedCool'] > testDF.loc['2019-09-25 19:05:00-07:00','TemperatureExpectedCool']:
            testDF.drop(testDF.loc['2019-09-25 19:00:00-07:00':'2019-09-25 20:00:00-07:00'])
            fullDF.loc['2019-09-25 19:00:00-07:00':'2019-09-25 20:00:00-07:00',:] = originalDF.loc['2019-09-25 19:00:00-07:00':'2019-09-25 20:00:00-07:00',:]
        else:
            pass
    except KeyError:
        pass      
    except ValueError: 
        pass
    try:
        if testDF.loc['2019-10-21 18:55:00-07:00','TemperatureExpectedCool'] > testDF.loc['2019-10-21 19:05:00-07:00','TemperatureExpectedCool']:
            testDF.drop(testDF.loc['2019-10-21 19:00:00-07:00':'2019-10-21 20:00:00-07:00'])
            fullDF.loc['2019-10-21 19:00:00-07:00':'2019-10-21 20:00:00-07:00',:] = originalDF.loc['2019-10-21 19:00:00-07:00':'2019-10-21 20:00:00-07:00',:]
        else:
            pass
    except KeyError:
        pass  
    except ValueError:
        pass
    
    try:
        testDF = pd.concat([testDF, fullDF.loc[[testDF.index[-1] + timedelta(minutes=5)]]]).sort_index()
    except KeyError:
        testDF = pd.concat([testDF, fullDF.loc[[testDF.index[-1] + timedelta(minutes=60)]]]).sort_index()

    testDF['time'] = testDF.index
    testDF['tdiff'] = testDF['time']-testDF['time'].shift()
    discontT = pd.concat([testDF.iloc[[0,-1]],testDF[testDF['tdiff'] > pd.Timedelta(5,'min')]]).sort_index()
    discontT = discontT[~discontT.index.duplicated(keep='first')]
    testDF = testDF[~testDF.index.duplicated(keep='first')]
    
    try:
        testDF.loc[:,'DR_event'] = True
    except ValueError:
        outliersList = np.concatenate([outliersList,np.array([identifier])])
        pd.DataFrame(outliersList).to_csv(finalDFPathway + 'outliers.csv')
        return outliersList
    
    finalDFHome = pd.DataFrame()

    for idx in range(discontT.shape[0]-1):  
        streakT = testDF.loc[discontT.iloc[idx].name:discontT.iloc[idx+1].name]
        streakT = streakT.drop(streakT.tail(1).index)
        

        na_free = streakT.dropna(subset=['Temperature_Ctrl', 'Temperature_Ctrl_lag_1','Temperature_Ctrl_lag_2','HourlyDryBulbTemperature',
                                         'HourlyDryBulbTemperature_lag_1','compCool_on_off_lag_1'])

        fullDF = pd.concat([fullDF, streakT[~streakT.index.isin(na_free.index)]])

        

        streakT = streakT.dropna(subset=['Temperature_Ctrl', 'Temperature_Ctrl_lag_1','Temperature_Ctrl_lag_2','HourlyDryBulbTemperature',
                                         'HourlyDryBulbTemperature_lag_1','compCool_on_off_lag_1'])


        fullDF = pd.concat([fullDF, streakT[streakT['HvacMode'] =='off']])


        streakT = streakT[streakT['HvacMode'] != 'off']
        try:
            streakT = streakT.sort_index()

        except KeyError and IndexError:
            print("EXCEPT")
            continue
            #this is the regression dataframe
        
        try: 
            count = streakT.index.tolist()
            Temperature_Ctrl_lag_1_expected = [streakT['Temperature_Ctrl_lag_1'][0]]
            Temperature_Ctrl_lag_2_expected = [streakT['Temperature_Ctrl_lag_2'][0]]
            HourlyDryBulbTemperature_lag_0 = streakT['HourlyDryBulbTemperature'].tolist()
            HourlyDryBulbTemperature_lag_1 = streakT['HourlyDryBulbTemperature_lag_1'].tolist()
    
                
            compCool_result_expected_lag_1 = [streakT['compCool_on_off_lag_1'][0]]
            TemperatureExpectedCool_expected = streakT['expectedSetPoint'].tolist()
            compCool_on_off_expected = []
            TemperatureCtrl_expected = []
            
            Temperature_Ctrl_lag_1_perfect = [streakT['Temperature_Ctrl_lag_1'][0]]
            Temperature_Ctrl_lag_2_perfect = [streakT['Temperature_Ctrl_lag_2'][0]]
            compCool_result_perfect_lag_1 = [streakT['compCool_on_off_lag_1'][0]]
            TemperatureExpectedCool_perfect = streakT['perfectResponse'].tolist()
            compCool_on_off_perfect = []
            TemperatureCtrl_perfect = []
            
            i = 0
            while i < len(count):
    
                temperatureCtrl_expected = regTemp.predict([[Temperature_Ctrl_lag_1_expected[i],Temperature_Ctrl_lag_2_expected[i],
                                                             HourlyDryBulbTemperature_lag_0[i],HourlyDryBulbTemperature_lag_1[i],
                                                             compCool_result_expected_lag_1[i]]])
                TemperatureCtrl_expected = TemperatureCtrl_expected + np.round(temperatureCtrl_expected,2).tolist()
                
                difference_expected = temperatureCtrl_expected - TemperatureExpectedCool_expected[i]
                
                #turn on 
    
                if compCool_result_expected_lag_1[i] == 0 and difference_expected[0] >= thresholdHome: 
                    compCool_on_off_expected_new = 1
                
                #stay on 
                elif compCool_result_expected_lag_1[i] ==  1 and difference_expected > 0:
                    compCool_on_off_expected_new = 1
                
                #turn off
                elif  compCool_result_expected_lag_1[i] == 1 and difference_expected <= 0:
                    compCool_on_off_expected_new = 0
                
                #stay off 
                else:
                    compCool_on_off_expected_new = 0            
                
                compCool_on_off_expected = compCool_on_off_expected + [compCool_on_off_expected_new]
                
    
                temperatureCtrl_perfect = regTemp.predict([[Temperature_Ctrl_lag_1_perfect[i],Temperature_Ctrl_lag_2_perfect[i],
                                                             HourlyDryBulbTemperature_lag_0[i],HourlyDryBulbTemperature_lag_1[i],
                                                             compCool_result_perfect_lag_1[i]]])
                                       
                TemperatureCtrl_perfect = TemperatureCtrl_perfect + np.round(temperatureCtrl_perfect,2).tolist()
                
                difference_perfect = temperatureCtrl_perfect - TemperatureExpectedCool_perfect[i]
                
                #turn on 
                if compCool_result_perfect_lag_1[i] == 0 and difference_perfect[0] >= thresholdHome: 
                    compCool_on_off_perfect_new = 1
                
                #stay on 
                elif compCool_result_perfect_lag_1[i] ==  1 and difference_perfect[0] > 0:
                    compCool_on_off_perfect_new = 1
                
                #turn off
                elif  compCool_result_perfect_lag_1[i] == 1 and difference_perfect[0] <= 0:
                    compCool_on_off_perfect_new = 0
                
                #stay off 
                else:
                    compCool_on_off_perfect_new = 0            
                
                compCool_on_off_perfect = compCool_on_off_perfect + [compCool_on_off_perfect_new]
                
                if i == len(count)-1:
                    newDF = pd.DataFrame({'DATE':count,'Temperature_Ctrl_lag_1_expected': Temperature_Ctrl_lag_1_expected, 'Temperature_Ctrl_lag_1_perfect': Temperature_Ctrl_lag_1_perfect,
                                          'Temperature_Ctrl_lag_2_expected': Temperature_Ctrl_lag_2_expected, 'Temperature_Ctrl_lag_2_perfect': Temperature_Ctrl_lag_2_perfect,
                                          'compCool_result_expected_lag_1':compCool_result_expected_lag_1, 'compCool_result_perfect_lag_1':compCool_result_perfect_lag_1,
                                          'compCool_on_off_expected':compCool_on_off_expected,'compCool_on_off_perfect':compCool_on_off_perfect,
                                          'Temperature_Ctrl_expected':TemperatureCtrl_expected, 'Temperature_Ctrl_perfect':TemperatureCtrl_perfect})
                    newDF['DATE'] = pd.to_datetime(newDF['DATE'],errors='coerce',utc=True).dt.tz_convert(str(timeZone))
                    newDF = newDF.set_index('DATE').sort_index()
                    
                    finalResultsDF = streakT.merge(newDF,how='right',left_index=True,right_index=True)
                    
                    finalDFHome = pd.concat([finalDFHome,finalResultsDF])
                    finalDFHome.to_csv(finalDFPathway + 'homesDR/' + str(identifier) + ".csv")
                    
                    i += 1
                    
    
                else:
                    compCool_result_expected_lag_1 = compCool_result_expected_lag_1 + [compCool_on_off_expected_new]
                    Temperature_Ctrl_lag_2_expected = Temperature_Ctrl_lag_2_expected + [Temperature_Ctrl_lag_1_expected[i]]
                    Temperature_Ctrl_lag_1_expected = Temperature_Ctrl_lag_1_expected + np.round(temperatureCtrl_expected,2).tolist()
                    
                    compCool_result_perfect_lag_1 = compCool_result_perfect_lag_1 + [compCool_on_off_perfect_new]
                    Temperature_Ctrl_lag_2_perfect = Temperature_Ctrl_lag_2_perfect + [Temperature_Ctrl_lag_1_perfect[i]]
                    Temperature_Ctrl_lag_1_perfect = Temperature_Ctrl_lag_1_perfect + np.round(temperatureCtrl_perfect,2).tolist()
                    
                    i += 1
            
            finalDFHome = finalDFHome.sort_index()
            finalDFHome.loc[:,'secondsExpected'] = np.where(finalDFHome['compCool_on_off_expected']==1,300,0)
            finalDFHome.loc[:,'secondsPerfect'] = np.where(finalDFHome['compCool_on_off_perfect']==1,300,0)
            
    
    
            finalDFHome.loc[:,'actual-expected'] = finalDFHome['compCool'] - finalDFHome['secondsExpected']
            finalDFHome.loc[:,'perfect-actual'] = finalDFHome['secondsPerfect'] - finalDFHome['compCool']
            finalDFHome.loc[:,'perfect-expected'] = finalDFHome['secondsPerfect'] - finalDFHome['secondsExpected']
            finalDFHome.to_csv(finalDFPathway + 'homesDR/' + str(identifier) + ".csv")
        
        except KeyError and IndexError:
            continue

    fullDF = fullDF[~fullDF.index.duplicated(keep='first')]
    fullDF = fullDF.sort_index()

    fullDF['perfectResponse'] = fullDF['TemperatureExpectedCool']
    fullDF['expectedResponse'] = fullDF['TemperatureExpectedCool']
        

    fullDF['actual-expected'] = 0
    fullDF['perfect-actual'] = 0
    fullDF['perfect-expected'] = 0
    fullDF['secondsExpected'] = fullDF['compCool']
    fullDF['secondsPerfect'] = fullDF['compCool']
        
        
    finalFullDF = pd.concat([fullDF, finalDFHome]).sort_index()
    finalFullDF.to_csv(finalDFPathway + 'homesFull/' + str(identifier) + ".csv")
    
    finalFullDFPrediction = finalFullDF
    return outliersList
main()