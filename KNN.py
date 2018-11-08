import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

############## DATA PARSERS MODULE #############################
def data_parser(data16_fname,data17_fname,loc_fname):
    '''
    This function will load the data in pandas dataframe and then
    remove the null entries.
    USAGE:
        INPUT:
            data16_fname   :  2016 pollution data filename
            data17_fname   :  2017 pollution data filename
            loc_fname  :  the location file
    '''
    #Reading the dataframe and cleaning them
    df16=pd.read_csv(data16_fname)
    _remove_nan_from_df(df16)

    #Reading the 2017 dataframe
    df17=pd.read_csv(data17_fname)
    _remove_nan_from_df(df17)

    #Rerading and cleaning the stations dataframe
    dfLoc=pd.read_csv(loc_fname)
    #_remove_nan_from_df(dfLoc)

    return df16,df17,dfLoc

def _remove_nan_from_df(df):
    '''
    This function will remove the nan values from the dataframe
    '''
    #Getting the mean of each column of full year
    mean_dict=df.describe().loc['mean'].to_dict()

    #Filling the missing values with the mean of each columns
    df.fillna(value=mean_dict,inplace=True)

    #Converting the daet sting to datetime object
    print "Converting the date string to datetime object"
    month=[]
    day=[]
    hour=[]
    for i in range(df.shape[0]):
        time=datetime.strptime(df.iloc[i]['date'],'%Y-%m-%d %X')
        month.append(time.month)
        day.append(time.day)
        hour.append(time.hour)

    df['month']=month
    df['day']=day
    df['hour']=hour

    print "Read and cleaned the dataframe\n",df.head()

############### PREDICTION FUNCTIONS ############################
def make_prediction(df16,df17,dfLoc):
    '''
    This function will find the K-Nearest Neighbour and then make
    prediction using the weighted average based on the distance from
    other neighbours.
    USAGE:
        INPUT:
            df16    : the pollution dataset for the year 2016
            df17    : the pollution dataset for the year 2017
            dfLoc   : the dataset containing the location of stations
    '''
    #iterating thought each of the entreis of 2017 dataset



if __name__=='__main__':
    data16_fname='dataset/madrid_2016.csv'
    data17_fname='dataset/madrid_2017.csv'
    loc_fname='dataset/stations.csv'

    #Cleaning the dataset
    data_parser(data16_fname,data17_fname,loc_fname)

    #Prediction function
