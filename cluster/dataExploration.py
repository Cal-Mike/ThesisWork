# Multi-Dimensional Profiling of Cyber Threats in Large-Scale Networks code
# Created by Michael Calnan
# Last edited: 14 SEP 2022

#import section
import math
from pickletools import long4
from re import I
from maxminddb import Reader
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
import geoip2.database, geoip2.errors
import socket, struct, random, os
from datetime import datetime,timedelta
import plotly.express as px
from data_features_type import DATA_FEATURES_DTYPE
import seaborn as sns
from ipaddress import ip_address
from maxminddb import Reader
import geoip2.database, geoip2.errors


#SET GLOBALS
FILENAME = 'features_table_0' #Set to filename of numpy file
CREATE_DATASET = False        #True for creating datasets, False for viewing the created dataset's information

CLUST_PATH_JUL = '/home/bdallen/work_new/2207/flows/'
CLUST_DATAFRAME_PATH = '/home/michael.calnan/calnan_thesis/cluster/dataframes/'

MAX_MIND_COUNTRY_PATH = '/home/michael.calnan/calnan_thesis/geolocation/GeoLite2-Country.mmdb'
MAX_MIND_ASN_PATH = '/home/michael.calnan/calnan_thesis/geolocation/GeoLite2-ASN.mmdb'
MAX_MIND_CITY_PATH = '/home/michael.calnan/calnan_thesis/geolocation/GeoLite2-City.mmdb'

##########
#READ DATA
##########

def to_storage(df, filename, path=CLUST_DATAFRAME_PATH):
    print('***putting it into a parquet file')
    df.to_parquet('{0}{1}.parquet.gzip'.format(path, filename),
        compression='gzip')
    print('***file upload complete')

def from_storage(filename, path=CLUST_DATAFRAME_PATH):
    return pd.read_parquet('{0}{1}.parquet.gzip'.format(path, filename))

def read_file(filename, path=CLUST_PATH_JUL):
    data_features_list = [item[0] for item in DATA_FEATURES_DTYPE]

    file_path = '{0}{1}.npy'.format(path, filename)
    npy_table = np.load(file_path)
    memoryview_table = memoryview(npy_table)
    np_table = np.frombuffer(memoryview_table, dtype=DATA_FEATURES_DTYPE)
    print('file{0} shape {1}'.format(file_path, np_table.shape, flush=True))

    df = pd.DataFrame(columns=data_features_list)
    file_length = np_table.shape[0]

    #for i in range(1000):
    for i in range(np_table.shape[0]):
        row = np_table[i]
        df.loc[i] = [row[item] for item in data_features_list]
    
    return df

###################
#DATA VISUALIZATION
###################

def plot_hist(df):
    '''
    Plots histograms for each dataframe feature.

    Parameters:
    df (Pandas.dataframe): dataframe to be plotted
    '''
    print('***plotting histogram')
    fig = plt.figure()
    df.hist(bins=50,figsize=(20,15))
    plt.show()
    plt.savefig('clust_hist_0to9.png')
    
    print('***histogram plotted')

def plot_corr(df):
    corr = df.corr().abs()
    mask = np.triu(np.ones_like(corr,dtype=bool))
    sns.heatmap(corr, mask=mask)

    plt.show()
    plt.savefig('clust_corr_0to9.png', bbox_inches='tight', pad_inches=0)


####################
#DATA TRANSFORMATION
####################
def ntoa(ip_list):
    result = []
    for ip in ip_list:
        try:
            #new_ip = ip_address(ip)
            #new_ip = socket.inet_ntoa(struct.pack('!I', ip))
            new_ip = socket.inet_ntoa(struct.pack('!L',(ip+2**32)%2**32))
        except struct.error:
            print('{0}, {1}'.format(ip, type(ip)))
            new_ip = '0.0.0.0'
        result.append(new_ip)
    return result

def add_geoinfo(df):
    '''
    Adds geolocation information to dataframe. Adds ASN and GeoName ID for each IP address. 0 (int) is used as placeholder for unidentified addresses.

    Parameters:
    df (Pandas.dataframe): dataframe containing 'ip_src' and ip_dst' columns

    Returns:
    Pandas.dataframe: Updated dataframe with additional columns for src and dst geolocation info.
    '''
    print('***adding geolocation transformation')

    print('Adding other_ASN column')
    ip_other_list = df['ip_other'].tolist()
    ip_other_ASN = ip_to_asn(ip_other_list)
    df['other_ASN'] = ip_other_ASN
    df['other_ASN'] = df['other_ASN'].astype('category')

    print('Adding other_COUNTRY')
    ip_other_COUNTRY = ip_to_country(ip_other_list)
    df['other_COUNTRY'] = ip_other_COUNTRY
    df['other_COUNTRY'] = df['other_COUNTRY'].astype('category')

    print('Adding other_LAT and other_LONG')
    other_lat,other_long = ip_to_latlong(ip_other_list)
    df['other_LAT'] = other_lat
    df['other_LONG'] = other_long

    print('***geolocation transformation completed')
    return df

def ip_to_asn(ip_list):
    '''
    Takes a list of IP addresses and returns a 1:1 list of ASN's.

    Parameters:
    ip_list (list<long>): IP addresses in LongInt format

    Returns:
    list<string>: ASN Number for each IP address. 0 for unidentified IP addresses. 
    '''
    result = []
    #ip_list = [socket.inet_ntoa(struct.pack('!I', ip)) for ip in ip_list]
    ip_list = ntoa(ip_list)

    with geoip2.database.Reader(MAX_MIND_ASN_PATH) as reader:
        for ip in ip_list:
            try:
                response = reader.asn(ip)
                result.append(response.autonomous_system_number)
            except geoip2.errors.AddressNotFoundError:
                result.append(0)
    return result
    
def ip_to_country(ip_list):
    '''
    Takes a list of IP addresses and returns a 1:1 list of geoname ID's.

    Parameters:
    ip_list (list<long>): IP addresses in LongInt format

    Returns:
    list<string>: Geoname ID's for each IP address. 0 for unidentified IP addresses. 
    '''
    result = []
    ip_list = ntoa(ip_list)
    #ip_list = [socket.inet_ntoa(struct.pack('!L', ip)) for ip in ip_list]
    
    with geoip2.database.Reader(MAX_MIND_COUNTRY_PATH) as reader:
        for ip in ip_list:
            try:
                response = reader.country(ip)
                result.append(response.country.name)
            except geoip2.errors.AddressNotFoundError:
                result.append('0')
    return result

def ip_to_latlong(ip_list):
    '''
    Takes a list of IP addresses and returns a 1:1 list of latitutde and longitude values.

    Parameters:
    ip_list (list<long>): IP addresses in LongInt format

    Returns:
    (list<float>, list<float>: Tuple of latitude and longitude for each IP address. 0.0 for unidentified IP addresses. 
    '''
    result_lat = []
    result_long = []
    #ip_list = [socket.inet_ntoa(struct.pack('!L', ip)) for ip in ip_list]
    ip_list = ntoa(ip_list)
    with geoip2.database.Reader(MAX_MIND_CITY_PATH) as reader:
        for ip in ip_list:
            try:
                response = reader.city(ip)
                result_lat.append(response.location.latitude)
                result_long.append(response.location.longitude)
            except geoip2.errors.AddressNotFoundError:
                result_lat.append(0.0)
                result_long.append(0.0)
    return (result_lat, result_long)

##############
#MAIN FUNCTION
##############

if __name__ == '__main__':
    start_time = datetime.now()

    if CREATE_DATASET == True:
        df = read_file(FILENAME)
        df_ = add_geoinfo(df)
        print(df_.head())
        to_storage(df_, FILENAME)
    else:
        df2 = from_storage(FILENAME)
        print(df2.head())
        print(df2.info())

    diff = datetime.now() - start_time
    print('Total runtime: {0}'.format(diff))