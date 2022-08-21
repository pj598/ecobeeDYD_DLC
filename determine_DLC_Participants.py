#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:02:25 2021

@author: pamelawildstein
"""

"""
figure out who is participating in DR events 
"""

import numpy as np  
import pandas as pd 
from datetime import timedelta

def main():
    counter = 0
    identifiers, export, finalList, timesDF, true_false_dr_df = createStuff()
    for identifier in identifiers:
        true_false_dr_df = checkDR(identifier, timesDF, true_false_dr_df, export)
        counter += 1 
        print(str(counter) + '/' + str(len(identifiers)))
        
    true_false_dr_df['8/21/19-2'] = np.where((true_false_dr_df['8/21/19-1'] == True) & (true_false_dr_df['8/21/19-2'] == True),False,true_false_dr_df['8/21/19-2'])

    true_false_dr_df['9/24/19-2'] = np.where((true_false_dr_df['9/24/19-1'] == True) & (true_false_dr_df['9/24/19-2'] == True),False,true_false_dr_df['9/24/19-2'])
    
    
    true_false_dr_df['DR_num'] = true_false_dr_df[['7/24/19','8/13/19-1','8/13/19-2','8/14/19','8/15/19',
                                               '8/21/19-1','8/21/19-2','8/26/19','8/27/19','8/28/19',
                                               '9/3/19','9/4/19','9/5/19','9/8/19','9/12/19','9/13/19',
                                               '9/24/19-1','9/24/19-2','9/25/19','10/16/19','10/21/19',
                                               '10/22/19']].sum(axis=1)
    true_false_dr_df['DR'] = true_false_dr_df['DR_num'] > 0
    true_false_dr_df.to_csv(export + 'dr_sce_2019FINALFINAL.csv')
    findParticipants(true_false_dr_df, export)
        
        
def createStuff():
    identifiersDF = pd.read_csv('') #identifiers you want to know if they participated in events 
    identifiers = identifiersDF['Identifier'].unique()
    
    timesDF = pd.read_csv('') #start and end times off events
    
    export = '' #export pathway
    
    finalList = pd.DataFrame()
    true_false_dr_df = pd.DataFrame(columns= ['Identifier','7/24/19','8/13/19-1','8/13/19-2','8/14/19','8/15/19',
                                               '8/21/19-1','8/21/19-2','8/26/19','8/27/19','8/28/19',
                                               '9/3/19','9/4/19','9/5/19','9/8/19','9/12/19','9/13/19',
                                               '9/24/19-1','9/24/19-2','9/25/19','10/16/19','10/21/19',
                                               '10/22/19'])
    
    return identifiers, export, finalList, timesDF, true_false_dr_df

def checkDR(identifier, timesDF, true_false_dr_df, export):
    print("CURRENT IDENTIFIER " + str(identifier))
    demandResponseDates = ['7/24/19','8/13/19-1','8/13/19-2','8/14/19','8/15/19',
                           '8/21/19-1','8/21/19-2','8/26/19','8/27/19','8/28/19',
                           '9/3/19','9/4/19','9/5/19','9/8/19','9/12/19','9/13/19',
                           '9/24/19-1','9/24/19-2','9/25/19','10/16/19','10/21/19',
                           '10/22/19']
    
    true_false_dr = pd.DataFrame({'Identifier':[str(identifier)]})
    
    df = pd.read_csv('',usecols=['DATE','CalendarEvent','HvacMode']) #dyd dataframe 
    df['DATE'] = pd.to_datetime(df['DATE'],errors='coerce',utc=True).dt.tz_convert(str('America/Los_Angeles'))
    df = df.set_index('DATE').sort_index()
    df = df.astype({'CalendarEvent':str})
    
    timesDF['START'] = pd.to_datetime(timesDF['START'],errors='raise',utc=True).dt.tz_convert(str('America/Los_Angeles'))
    timesDF['END'] = pd.to_datetime(timesDF['END'],errors='raise',utc=True).dt.tz_convert(str('America/Los_Angeles'))
    
    for date, timeStart, end in zip(demandResponseDates, timesDF['START'], timesDF['END']): 
        timeEnd = end - timedelta(minutes=10)
        
        try:
            true_false_dr[str(date)] = df.loc[timeStart,'CalendarEvent'].isdecimal()
            if df.loc[timeStart,'CalendarEvent'].isdecimal() == True :
                dfEvent = df.loc[timeStart:timeEnd]
                modes = ['cool','auto']
                dfEvent = dfEvent[dfEvent.HvacMode.isin(modes)]
  
                if len(dfEvent.index) ==  0 :
                    true_false_dr[str(date)] = False 
                
                else:
                    pass
            else: 
                pass
            
        except KeyError: 
            true_false_dr[str(date)] = False

            
    true_false_dr_df = pd.concat([true_false_dr_df, true_false_dr])
    true_false_dr_df.to_csv(export + 'participate_true_false.csv')
    
    return true_false_dr_df

def findParticipants(true_false_dr_df, export):
    trueIdentifiersDF = true_false_dr_df[true_false_dr_df['DR'] > 0]
    trueIdentifiers = trueIdentifiersDF['Identifier'].tolist()
    
    finalList = pd.DataFrame({'Identifier': trueIdentifiers})
    finalList.to_csv(export + 'DLC_participants.csv')
    

main()
    
    
  