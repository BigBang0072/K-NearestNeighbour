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
            loc_fname      :  the location file
    '''
    try:
        print "Loading the cached data"
        df16=pd.read_csv('dataset/cleaned_2016.csv')
        df17=pd.read_csv('dataset/cleaned_2017.csv')
        dfLoc=pd.read_csv(loc_fname)
        dfLoc.rename({'id':'station'},axis='columns',inplace=True)

        return df16,df17,dfLoc
    except:
        print ("Load Failed, cleaning the dataset")
        #Reading the dataframe and cleaning them
        df16=pd.read_csv(data16_fname)
        _remove_nan_from_df(df16)

        #Reading the 2017 dataframe
        df17=pd.read_csv(data17_fname)
        _remove_nan_from_df(df17)

        #Rerading and cleaning the stations dataframe
        dfLoc=pd.read_csv(loc_fname)
        dfLoc.rename({'id':'station'},axis='columns',inplace=True)
        #_remove_nan_from_df(dfLoc)

        df16.to_csv('dataset/cleaned_2016.csv')
        df17.to_csv('dataset/cleaned_2017.csv')

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
    year=[]
    for i in range(df.shape[0]):
        time=datetime.strptime(df.iloc[i]['date'],'%Y-%m-%d %X')
        year.append(time.year)
        month.append(time.month)
        day.append(time.day)
        hour.append(time.hour)

    df['year']=year
    df['month']=month
    df['day']=day
    df['hour']=hour

    #Sorting the dataframe
    print "Sorting the dataframe according to year-month-day-hour"
    sort_by=['year','month','day','hour']
    df.sort_values(sort_by,inplace=True)

    print "Read and cleaned the dataframe\n",df.head()

############### PREDICTION FUNCTIONS ############################
def make_prediction(k_val,df16,df17,dfLoc,type):
    '''
    This function will find the K-Nearest Neighbour and then make
    prediction using the weighted average based on the distance from
    other neighbours.
    USAGE:
        INPUT:
            k_val   : the number of nearest neighbour we have to take
            df16    : the pollution dataset for the year 2016
            df17    : the pollution dataset for the year 2017
            dfLoc   : the dataset containing the location of stations
            type    : recursive or rolling
    '''
    #Taking out the section of 2017 for prediction
    df17=df17[ (df17['month']==1) & (df17['day']==1) ]

    #Joining the pollution dataframe with the location info
    df16=pd.merge(df16,dfLoc,how='left',on='station')
    df17=pd.merge(df17,dfLoc,how='left',on='station')

    print "df16 description\n",df16.describe()
    print "df17 description\n",df17.describe()

    #iterating over 2017 entreis for making prediction
    errors=[]

    for i in range(df17.shape[0]):
        #Taking out the 2017 entry separately
        pred_series=df17.iloc[i]
        print "\nMaking prediction for df17 entry: ",i
        print "Sanity check",df16.shape
        #print pred_series

        #Getting the prediction
        error=make_recursive_prediction(k_val,df16,pred_series,norm=False)
        errors.append(error)

        #handling the type of the rolling window
        if(type=='recursive'):
            #Adding the current series to our record
            df16=df16.append(pred_series,ignore_index=True)
        else:
            #Dropping the ith element (reindexing not done)
            df16.drop([i],axis=0,inplace=True)#removing one element at a time
            #Adding the new element
            df16=df16.append(pred_series,ignore_index=True)

    return errors

def make_recursive_prediction(k_val,df16,pred_series,norm=False):
    '''
    This function will calculation the distance from the neighbour
    and then choose the best top k among them to calculate the average
    prediction.
    '''
    print "Calculating the distance from the whole previous dataframe"
    #Getting the distance from all the element (way 1)
    # dist_df=df16.apply(_get_distance_unnormalized,
    #             axis=1,
    #             pred_series=pred_series,
    #             dfLoc=dfLoc,
    #             norm=norm)

    #Getting the distance from all the other elements (way2)
    extract_elements=['lon','lat','elevation',      #spatial elemets
                    'hour','day','month',#'year',#temporal and features
                    'EBE','NMHC','NO','NO_2','O_3',
                    'SO_2','TCH','TOL','BEN','CO'
                    ]
    #Converting the dataframe into matrix with the required element
    df16_mat=(df16[extract_elements]).values
    pred_vec=(pred_series[extract_elements]).values
    #Extracting the prediction feature in same order
    pred_mat=(df16[['PM10','PM25']]).values

    #Now making the prediction vectorially
    distance=_get_distance_unnormalized_V2(df16_mat,pred_vec,norm)

    #Now taking the top k to make prediction
    pred_PM10=0.0
    pred_PM25=0.0
    norm_distance=0.0

    done_idx=[]
    for i in range(k_val):
        min=1e6 #setting a default value (INTMAX)
        min_idx=-1
        #Calculating the minimum (doing iteratively instead of sorting)
        for idx in range(distance.shape[0]):
            if((distance[idx]<min) and (idx not in done_idx) ):
                min=distance[idx]
                min_idx=idx
        #Saving the minimum to the list for taking distinct next time
        done_idx.append(min_idx)

        #calculating the prediction
        pred_PM10+=(1.0/distance[min_idx])*pred_mat[min_idx,0]
        pred_PM25+=(1.0/distance[min_idx])*pred_mat[min_idx,1]
        norm_distance+=(1.0/distance[min_idx])

    #Averaging the weighted sum
    act_PM10=pred_series['PM10']
    act_PM25=pred_series['PM25']
    pred_PM10=pred_PM10/norm_distance
    pred_PM25=pred_PM25/norm_distance

    #Printing the error and the prediction
    #print done_idx
    print "PM10 actual:{} pred:{} error:{}".format(act_PM10,
                                                    pred_PM10,
                                                    act_PM10-pred_PM10)
    print "PM25 actual:{} pred:{} error:{}".format(act_PM25,
                                                    pred_PM25,
                                                    act_PM25-pred_PM25)

    #Creating the result tuple
    result=(act_PM10-pred_PM10,act_PM25-pred_PM25)

    return result

def _get_distance_unnormalized(data_series,pred_series,dfLoc,norm):
    '''
    This function will calculate the distance between the two points
    in the location-pollution space.
    '''
    #Calculating the distance between station (spatial distance)
    stat1_idx=data_series.loc['station']
    stat2_idx=pred_series.loc['station']
    #Extracting the location data from locdf
    locinfo1=dfLoc.loc[stat1_idx][['lon','lat','elevation']]
    locinfo2=dfLoc.loc[stat2_idx][['lon','lat','elevation']]
    #Add optimal normalization here
    if norm==True:
        locinfo1=locinfo1/np.sum(np.square(locinfo1))
        locinfo2=locinfo2/np.sum(np.square(locinfo2))
    #Getting the distance between them
    diff1=np.sum(np.square(locinfo1.subtract(locinfo2)))

    #Calculating the distance between event (temporal and feature)
    extract_entry=['hour','day','month','BEN','CO',
                    'EBE','NMHC','NO','NO_2','O_3',
                    'SO_2','TCH','TOL'
                    ]
    #Extracting out the info
    datainfo=data_series[extract_entry]
    predinfo=pred_series[extract_entry]
    #Normlaizing them optinally
    if norm==True:
        datainfo=datainfo/np.sum(np.square(datainfo))
        predinfo=predinfo/np.sum(np.square(predinfo))
    #Subtracting the value
    diff2=np.sum(np.square(datainfo.subtract(predinfo)))

    total_distance=diff2+diff1
    PM10=data_series['PM10']
    PM25=data_series['PM25']

    #Returnig these three items incase the order is not maintained
    return PM10,PM25,total_distance

def _get_distance_unnormalized_V2(data_mat,pred_vec,norm):
    '''
    This function will calculate the distance using only the matrix
    miltiplication to make the computation fast
    '''
    #Normlaizing the matrix if required
    if norm==True:
        data_mat=data_mat/np.sum(np.square(data_mat),axis=1,keepdims=True)
        pred_vec=pred_vec/np.sum(np.square(pred_vec))

    #Now finding the distance vectorially
    distance=np.sum(np.square(data_mat-pred_vec),axis=1)**(0.5)

    return distance

##################### RESULTS PLOTTING ########################
def plot_predictions(errors):
    '''
    This fucntion will take the list tuple of errror and plot a
    histogram.
    '''
    #Extracting out the errors from the list of tuples
    error_PM10,error_PM25=zip(*errors)

    #Creating the figure object
    fig=plt.figure()
    nbins=100

    #Adding the histogram for the first element
    ax1=fig.add_subplot(121)
    ax1.hist(error_PM10,bins=nbins,facecolor='green',edgecolor='black',alpha=0.67)
    ax1.set_title("Error Histogram PM10 prediction")
    ax1.set_xlabel("Absolute Error")
    ax1.set_ylabel("Frequency")

    ax2=fig.add_subplot(122)
    ax2.hist(error_PM25,bins=nbins,facecolor='green',edgecolor='black',alpha=0.67)
    ax2.set_title("Error Histogram PM25 prediction")
    ax2.set_xlabel("Absolute Error")
    ax2.set_ylabel("Frequency")

    plt.show()

if __name__=='__main__':
    data16_fname='dataset/madrid_2016.csv'
    data17_fname='dataset/madrid_2017.csv'
    loc_fname='dataset/stations.csv'
    k_val=10

    #Cleaning the dataset
    df16,df17,dfLoc=data_parser(data16_fname,data17_fname,loc_fname)

    #Prediction function
    errors=make_prediction(k_val,df16,df17,dfLoc,type='rolling')

    #Plotting the results
    plot_predictions(errors)
