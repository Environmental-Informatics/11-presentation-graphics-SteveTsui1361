#!/bin/env python
# Add your own header comments
# This code is used to do the work of presentation graphics

import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # Replace negative values
    DataDF['Discharge'] = DataDF['Discharge'].mask(DataDF['Discharge']<0, np.nan)
    
    # quantify the number of missing values
    #MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF)

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    #Clip data within the time period
    DataDF.index = pd.to_datetime(DataDF.index)
    mask = (DataDF.index >= startDate) & (DataDF.index <= endDate)
    DataDF = DataDF.loc[mask]
    
    # quantify the number of missing values
    #MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF )

def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    # open and read the file
    DataDF = pd.read_csv(fileName)
    DataDF = DataDF.set_index('Date')
    
    return( DataDF )


# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    # define filenames as a dictionary
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    fileName1 = { "Annual": "Annual_Metrics.csv",
                 "Monthly": "Monthly_Metrics.csv" }
    
    color = {"Wildcat":'b',
             "Tippe":'orange'}
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    #MissingValues = {}
    
    
    for file in fileName.keys():
        
        DataDF[file] = ReadData(fileName[file])
        
        # clip to consistent period
        DataDF[file] = ClipData( DataDF[file], '2014-10-01', '2019-09-30' )
            
        #print(DataDF[file].head())
            
        plt.plot(DataDF[file]['Discharge'], label=riverName[file], color=color[file])
    plt.legend()
    plt.title('Daily Discharge Hydrograph')
    plt.ylabel('Discharge (cfs)')
    plt.xlabel('Time')
    plt.savefig('Daily_Hydro.png', dpi=96)    
    plt.show()
       
    # Create annual dataframe    
    DataDF['Annual'] = ReadMetrics(fileName1['Annual'])
        
    # clip to consistent period
    #DataDF['Annual'] = ClipData( DataDF['Annual'], '2014-10-01', '2018-10-01' )
        
    #Groupby data by the station name
    Annual_Wildcat = DataDF['Annual'].groupby('Station')
    
    #Plot annual TQmean
    for name, data in Annual_Wildcat:
        plt.plot(data.index.values, data.TQmean.values, label=riverName[name], color=color[name])
    plt.legend()
    #plt.gca().xaxis_date()
    #plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
    #plt.tight_layout()
    plt.title('TQmean')
    plt.ylabel('TQmean')
    plt.xlabel('Time')
    plt.savefig('TQmean.png', dpi=96)
    plt.show()
    
    #Plot annual coefficient variation
    for name, data in Annual_Wildcat:
        plt.plot(data.index.values, data['Coeff Var'].values, label=riverName[name], color=color[name])
    plt.legend()
    plt.title('Coefficient Variation')
    plt.ylabel('Coefficient Vaviation')
    plt.xlabel('Time')
    plt.savefig('Coeff_Var.png', dpi=96)
    plt.show()
    
    #Plot annual R-B Index
    for name, data in Annual_Wildcat:
        plt.plot(data.index.values, data['R-B Index'].values, label=riverName[name], color=color[name])
    plt.legend()
    plt.title('R-B Index')
    plt.ylabel('R-B Index')
    plt.xlabel('Time')
    plt.savefig('R-B_Index.png', dpi=96)
    plt.show()
   
    # Create monthly dataframe    
    DataDF['Monthly'] = ReadMetrics(fileName1['Monthly'])
        
    # clip to consistent period
    #DataDF['Monthly'] = ClipData( DataDF['Monthly'], '2014-10-01', '2018-10-01' )
    
    MoDataDF = DataDF['Monthly'].groupby('Station')
    
    for name, data in MoDataDF:
        cols=['Mean Flow']
        m=[3,4,5,6,7,8,9,10,11,0,1,2]
        index=0
        
        # Create a new dataframe
        MonthlyAverages=pd.DataFrame(0,index=range(1,13),columns=cols)
        
        # Create the output table
        for i in range(12):
            MonthlyAverages.iloc[index,0]=data['Mean Flow'][m[index]::12].mean()
            index+=1
        plt.plot(MonthlyAverages.index.values, MonthlyAverages['Mean Flow'].values, label=riverName[name], color=color[name])    
    plt.legend()
    plt.title('Average Annual Monthly Flow')
    plt.ylabel('Discharge (cfs)')
    plt.xlabel('Time')
    plt.savefig('Annual_mo.png', dpi=96)
    plt.show()
    
    #Return Period of annual peak flow
    DataDF['Annual'] = DataDF['Annual'].drop(columns=['site_no','Mean Flow','TQmean','Median','Coeff Var','Skew','R-B Index','7Q','3xMedian'])
    Annual_peak = DataDF['Annual'].groupby('Station')
    
    # Plot return period 
    for name, data in Annual_peak:
        # Sort the data in descnding order
        flow = data.sort_values('Peak Flow', ascending=False)
        # Check the data
        #print(flow.head())
        # Calculate the rank and reversing the rank to give the largest discharge rank of 1
        ranks1 = stats.rankdata(flow['Peak Flow'], method='average')
        ranks2 = ranks1[::-1]
        
        # Calculate the exceedance probability
        exceed_prob = [100*(ranks2[i]/(len(flow)+1)) for i in range(len(flow))]
        #print(exceed_prob)
        
        #plt.plot(data.index.values, data.TQmean.values, label=riverName[name], color=color[name])
        #plt.figure()
        #Plot scatter(prob,flow)
        plt.plot(exceed_prob, flow['Peak Flow'],label=riverName[name], color=color[name])
        # Add gird lines to both axes
    plt.grid(which='both')
    plt.legend()
    plt.title('Return Period of Annual Peak Flow Events')
    plt.ylabel('Peak Discharge (cfs)')
    plt.xlabel('Exceedance Probability (%)')
    plt.xticks(range(0,100,5))
    plt.savefig('Exceed_prob.png', dpi=96)
    plt.show()
    plt.rcParams.update({'font.size':30})