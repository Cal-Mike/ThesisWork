
# Multi-Dimensional Profiling of Cyber Threats in Large-Scale Networks code
# Created by Michael Calnan
# Last edited: 14 SEP 2022

#import section
from locale import normalize
from re import I
from maxminddb import Reader
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
import geoip2.database, geoip2.errors
import socket, struct, random, os
from datetime import datetime,timedelta
import plotly.express as px
import seaborn as sns

#SET GLOBALS
CLASS_PATH_JUL = '/home/bdallen/work_new/2207/labeled_packets/'
CLASS_DATASET_TRAINING_PATH = '/home/michael.calnan/calnan_thesis/classifier/datasets/training/'
CLASS_DATASET_VAL_PATH = '/home/michael.calnan/calnan_thesis/classifier/datasets/validation/'

MAX_MIND_COUNTRY_PATH = '/home/michael.calnan/calnan_thesis/geolocation/GeoLite2-Country.mmdb'
MAX_MIND_ASN_PATH = '/home/michael.calnan/calnan_thesis/geolocation/GeoLite2-ASN.mmdb'
MAX_MIND_CITY_PATH = '/home/michael.calnan/calnan_thesis/geolocation/GeoLite2-City.mmdb'

FLOW_TYPE_INSIDE_ONLY = 0   #flow_type label for outgoing blocked traffic
FLOW_TYPE_OUTSIDE_ONLY = 1  #flow_type label for incoming blocked traffic
FLOW_TYPE_OUTGOING = 2      #flow_type label for outgoing allowed traffic
FLOW_TYPE_INCOMING = 3      #flow_type label for incoming allowed traffic
FLOW_TYPE_UNIDENTIFIED = 4  #flow_type label for unknown origin or firewall action

FLOW_DICT = {1: 'INCOMING_ALLOWED',
    2: 'OUTGOING_ALLOWED',
    3: 'INSIDE_BLOCKED',
    4: 'INSIDE_NOT_ALLOWED',
    5: 'INSIDE_DROPPED_INBOUND',
    6: 'INSIDE_DROPPED_OUTBOUND',
    7: 'OUTSIDE_BLOCKED',
    8: 'OUTSIDE_NOT_ALLOWED',
    9: 'OUTSIDE_DROPPED_INBOUND',
    10: 'OUTSIDE_DROPPED_OUTBOUND'}

##########
#READ DATA
##########

def read_file():
    '''
    Reads a single random parquet file in PATH directory into a single dataframe.

    Returns:
    Pandas.dataframe: Contains parquet file information.
    '''
    dir_list = os.listdir(CLASS_PATH_JUL)

    # filename = [name for name in dir_list if 't_2022-07-14_13' in name][0] #Training Set
    # filename = [name for name in dir_list if 't_2022-07-14_14_00' in name][0] #Training Set
    # filename = [name for name in dir_list if 't_2022-07-14_14_15' in name][0] #Training Set
    # filename = [name for name in dir_list if 't_2022-07-14_14_30' in name][0] #Training Set
    filename = [name for name in dir_list if 't_2022-07-14_14_45' in name][0] #Training Set 
    #filename = [name for name in dir_list if 't_2022-07-14_15' in name][0] #Training Set

    # filename = [name for name in dir_list if 't_2022-07-15_14' in name][0] #Validation Set
    # filename = [name for name in dir_list if 't_2022-07-16_14' in name][0] #Validation Set
    # filename = [name for name in dir_list if 't_2022-07-17_14' in name][0] #Validation Set
    # filename = [name for name in dir_list if 't_2022-07-18_14' in name][0] #Validation Set
    # filename = [name for name in dir_list if 't_2022-07-19_14' in name][0] #Validation Set
    # filename = [name for name in dir_list if 't_2022-07-20_10' in name][-1] #Validation Set

    df_table = pd.read_parquet(CLASS_PATH_JUL + filename)
    return df_table, filename[:-5]
    
def to_storage(df, filename):
    print('***putting it into a parquet file')

    df.to_parquet('{0}class_dataset_{1}.parquet.gzip'.format(CLASS_DATASET_TRAINING_PATH, filename),
        compression='gzip')
 
    print('***file upload complete')

def from_storage(name):
    return pd.read_parquet('{0}class_dataset_{1}.parquet.gzip'.format(CLASS_DATASET_TRAINING_PATH,name))


#################
#DATA EXPLORATION
#################

def ip_to_asn(ip_list):
    '''
    Takes a list of IP addresses and returns a 1:1 list of ASN's.

    Parameters:
    ip_list (list<long>): IP addresses in LongInt format

    Returns:
    list<string>: ASN Number for each IP address. 0 for unidentified IP addresses. 
    '''
    result = []
    ip_list = [socket.inet_ntoa(struct.pack('!L', ip)) for ip in ip_list]
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
    list<string>: ISO Code for each IP address. '0' for unidentified IP addresses. 
    '''
    result = []
    ip_list = [socket.inet_ntoa(struct.pack('!L', ip)) for ip in ip_list]
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
    ip_list = [socket.inet_ntoa(struct.pack('!L', ip)) for ip in ip_list]
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


####################
#DATA TRANSFORMATION
####################

def add_geoinfo(df):
    '''
    Adds geolocation information to dataframe. Adds ASN and GeoName ID for each IP address. 0 (int) is used as placeholder for unidentified addresses.

    Parameters:
    df (Pandas.dataframe): dataframe containing 'ip_src' and ip_dst' columns

    Returns:
    Pandas.dataframe: Updated dataframe with additional columns for src and dst geolocation info.
    '''
    print('***adding geolocation transformation')

    print('Adding src_ASN column')
    ip_src_list = df['ip_src'].tolist()
    ip_src_ASN = ip_to_asn(ip_src_list)
    df['src_ASN'] = ip_src_ASN

    print('Adding src_COUNTRY')
    ip_src_COUNTRY = ip_to_country(ip_src_list)
    df['src_COUNTRY'] = ip_src_COUNTRY
    df['src_COUNTRY'] = df['src_COUNTRY'].astype('category')

    print('Adding dst_ASN column')
    ip_dst_list = df['ip_dst'].tolist()
    ip_dst_ASN = ip_to_asn(ip_dst_list)
    df['dst_ASN'] = ip_dst_ASN

    print('Adding dst_COUNTRY')
    ip_dst_COUNTRY = ip_to_country(ip_dst_list)
    df['dst_COUNTRY'] = ip_dst_COUNTRY
    df['dst_COUNTRY'] = df['dst_COUNTRY'].astype('category')

    print('Adding src_LAT and src_LONG')
    src_lat,src_long = ip_to_latlong(ip_src_list)
    df['src_LAT'] = src_lat
    df['src_LONG'] = src_long

    print('Adding dst_LAT and dst_LONG')
    dst_lat,dst_long = ip_to_latlong(ip_dst_list)
    df['dst_LAT'] = dst_lat
    df['dst_LONG'] = dst_long

    print('***geolocation transformation completed')
    return df

##############
#MAIN FUNCTION
##############

if __name__ == '__main__':
    start_time = datetime.now()

    #df = from_storage()
    #df = read_directory()
    df, filename = read_file()
    print(filename)
    #df = df.head(1000)
    print(df)

#DATA TRANSFORMATIONS
    df = add_geoinfo(df)
    to_storage(df, filename)

#DATA EXPLORATION
    #feature_exploration(df)
    df['src_COUNTRY_enc'] = LabelEncoder().fit_transform(df['src_COUNTRY'])
    df['dst_COUNTRY_enc'] = LabelEncoder().fit_transform(df['dst_COUNTRY'])
    print(df.corr()['packet_label'].sort_values(ascending=False, key=abs))
    print(df['packet_label'].value_counts(normalize=True, ascending=False))
    print(df['src_COUNTRY'].unique())
    print(df['dst_COUNTRY'].unique())


    diff = datetime.now() - start_time
    print('Total runtime: {0}'.format(diff))