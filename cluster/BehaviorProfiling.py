# Multi-Dimensional Profiling of Cyber Threats in Large-Scale Networks code
# Created by Michael Calnan
# Last edited: 14 SEP 2022

########## IMPORTS  ##########
from cProfile import label
from itertools import count
from operator import itemgetter
import os, socket, struct
import pickle
from re import I
from unicodedata import numeric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from keras.utils import np_utils
import tensorflow as tf
from category_encoders.binary import BinaryEncoder
from data_features_type import DATA_FEATURES_DTYPE
import plotly.express as px
from pandas.core.common import flatten
from scipy.cluster.vq import vq
from re import search


########## GLOBAL VARIABLES ##########

CLUST_DATAFRAME_PATH = '/home/michael.calnan/calnan_thesis/cluster/dataframes/'
FEATURE_NAMES = ['flow_types', 'ip_other', 'other_ASN', 'other_COUNTRY', 'other_LAT', 'other_LONG',
                'list_in_packets', 'list_in_bytes', 'list_in_mean_packet_length', 'list_in_packets_per_second', 'list_in_bytes_per_second', 'list_out_bytes_per_second',
                'list_out_packets', 'list_out_bytes', 'list_out_mean_packet_length', 'list_out_packets_per_second', 'list_out_bytes_per_second', 'list_out_bytes_per_second',
                'list_t_duration']

BEHAVIOR_TABLE = {
        0: 'None', 1: 'Google Search', 2: 'Microsoft Products', 3: 'Cloud Computing', 4: 'Media',
        5: 'Media', 6: 'None', 7: 'None', 8: 'Microsoft Products', 9: 'None',
        10: 'None', 11: 'None', 12: 'Media', 13: 'None', 14: 'Cloud Computing',
        15: 'Microsoft Products', 16: 'None', 17: 'None', 18: 'None', 19: 'Microsoft Products',
        20: 'Google Search', 21: 'None', 22: 'None', 23: 'None', 24: 'Media'
    }

########## FUNCTIONS ##########
def from_storage(filename, path=CLUST_DATAFRAME_PATH):
    return pd.read_parquet('{0}{1}'.format(path, filename))

def to_storage(df, filename, path=CLUST_DATAFRAME_PATH):
    print('***putting it into a parquet file')
    df.to_parquet('{0}{1}.parquet.gzip'.format(path, filename),
        compression='gzip')
    print('***file upload complete')

def ntoa(ipa):
    try:
        #new_ip = ip_address(ip)
        #new_ip = socket.inet_ntoa(struct.pack('!I', ip))
        new_ip = socket.inet_ntoa(struct.pack('!L',(ipa+2**32)%2**32))
    except struct.error:
        print('{0}, {1}'.format(ipa, type(ipa)))
        new_ip = '0.0.0.0'
    return new_ip

def resolveIP(ip_str):
    try:
        result = socket.gethostbyaddr(ip_str)[0]
    except:
        result = 'UNKNOWN'
    return result

def resolveUnknownIP(row):
    value = row['IP']

    if  row['URL'] != 'UNKNOWN':
        return row['URL']

    if '20.190.132.' in value or '20.60.128.' in value or '13.69.116.' in value or '13.89.179.' in value or '52.112.95.' in value or '20.40.229.' in value or '52.239.' in value or '52.96.' in value or '52.111.245.2' in value or '52.115.' in value or '40.97.204' in value or '52.245.136' in value or '40.97.205.' in value or '20.140.232.' in value or '40.96.60.' in value or '13.107.6.163' in value:
        result = 'Microsoft'
    elif '52.113.207.' in value or '52.111.244.' in value or '52.114.132.' in value or '52.113.207' or '52.109.0.' in value:
        result = 'Microsoft'
    elif '72.21.91.66' in value or '192.229.163.25' in value or '199.127.204.' in value or '17.157.64.' in value or '93.184.215.201' in value:
        result = 'Media'
    elif '159.53.232.35' in value:
        result = 'Banking'
    elif '205.77.96.' in value or '156.112.108.' in value:
        result = 'DoD E-mail'
    elif '146.75.95.' in value:
        result = 'Fastly Cloud Computing'
    elif '170.72.135.' in value:
        result = 'Video Call'
    elif '75.75.77.' in value or '203.209.194.' in value:
        result = 'ISP'
    elif '150.136.26.' in value:
        result = 'Oracle Public Cloud'
    else:
        result = None
    return result

def expandDataset(df):
    df_cluster = pd.DataFrame(df['lists'].to_list())
    for feature in FEATURE_NAMES[4:]:
        df_cluster[feature] = df[feature]
    df_cluster['other_ASN'] = df['other_ASN']
    return df_cluster

def clean_dataframe(df):
    '''Remove erronous columns from dataset.
    '''
    df = df[df.label != 'None']

    #drop ip addresses and index columns
    if 'ip_src' in df.columns and 'ip_dst' in df.columns:
        df = df.drop(labels=['ip_src','ip_dst'],axis=1,inplace=False)
    if 'index' in df.columns:
        df = df.drop(labels=['index'], axis=1,inplace=False)

    # if 'lists' in df.columns: #USE FOR RFC
    #     df = df.drop(labels=['lists'], axis=1, inplace=False) #USE FOR RFC
    return df

def dataset_to_matrix(dataset, target_encoder=None):
    '''Encode target feature array to one-hot encoding.
    '''
    #If no encoder for labels, create a new encoder
    if target_encoder is None:
        target_encoder = LabelEncoder()
        target_encoder.fit(dataset)

    dataset_ = target_encoder.transform(dataset) #encode dataset
    return np_utils.to_categorical(dataset_, num_classes=4).astype(np.uint8), target_encoder #create categorical matrix
    
def matrix_to_dataset(matrix, encoder):
    '''Encode target feature one-hot encoding into a label encoding array.
    '''
    matrix_ = np.argmax(matrix, axis=1)
    return encoder.inverse_transform(matrix_)

def split_dataframe(df):
    '''Create target and feature vector arrays.
    '''    
    df_target = df['label']
    df.pop('label')
    return df, df_target

def encode_dataframe(df, encoder=None):
    '''Encode entire dataset to values between 0 and 1.
    '''
    
    if encoder is None:
        with open('encoders/ct.ColumnTransformer', 'rb') as numerical_file:
            encoder = pickle.load(numerical_file)

    df_encoded = encoder.transform(df)

    return df_encoded, encoder

def simple_ANN(X_train, y_train_matrix, X_val, y_val_matrix):
    '''Create Multi-Layer Perceptron.
    '''
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[0:])),
    #     tf.keras.layers.Dropout(0.25),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dropout(0.25),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dropout(0.25),
    #     tf.keras.layers.Dense(4, activation='softmax')
    # ])
    X_train = tf.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = tf.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,input_shape=[X_train.shape[0], 1]),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    print(model.summary())

    #Compile model metrics and training hyperparameters
    model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=['accuracy',
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision()])
    
    #Create early stopping callback function
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', #use validation loss
        patience=10, #stop after 10 non-improving empochs
        restore_best_weights=True
    )
    
    col_sum = y_train_matrix.sum(axis=0) #Get sum of each label
    FLOW0 = col_sum[0]
    FLOW1 = col_sum[1]
    FLOW2 = col_sum[2]
    FLOW3 = col_sum[3]

    s = FLOW0 + FLOW1 + FLOW2 + FLOW3 #Create weights from ratio of class membership to overall samples
    w0 = s / FLOW0
    w1 = s / FLOW1
    w2 = s / FLOW2
    w3 = s / FLOW3

    classweight = {0: w0, 1: w1, 2: w2, 3: w3}

    #Train model
    model.fit(
            x=X_train,
            y=y_train_matrix,
            batch_size=512,
            validation_data=(X_val, y_val_matrix),
            verbose=1,
            epochs=50, 
            callbacks=[es]
            #class_weight=classweight
        )

    return model
########### MAIN METHODS ###############

def readDataset(path=CLUST_DATAFRAME_PATH, number=None):
    drop_features_list = [item[0] for item in DATA_FEATURES_DTYPE if item[0] not in FEATURE_NAMES]

    print('...reading files')
    dfList = []

    if number is not None:
        print('file {0}'.format(number))
        file = 'features_table_{0}.parquet.gzip'.format(number)
        df = from_storage(file, path)
        df = df.drop(columns=drop_features_list)

        df['lists'] = df['flow_types'].values.tolist()
        df['lists'] = df['lists'].apply(lambda x: list(flatten(x)))

        df.drop(['flow_types'], axis=1, inplace=True)
        df = df.dropna().reset_index(drop=True)

        print(df)
        return df
    else:

        for i in range(9,20):
            print('file {0}'.format(i))
            file = 'features_table_{0}.parquet.gzip'.format(i)
            df = from_storage(file, path)
            df = df.drop(columns=drop_features_list)


            df['lists'] = df['flow_types'].values.tolist()
            df['lists'] = df['lists'].apply(lambda x: list(flatten(x)))


            df.drop(['flow_types'], axis=1, inplace=True)
            dfList.append(df)
        print('...creating dataset')
        df = pd.concat(dfList, ignore_index=True)
        df = df.dropna().reset_index(drop=True)

        print(df)

        return df

def createBehaviorTable():
    #df, scaler = readDataset()
    df = readDataset()
    
    df_cluster = pd.DataFrame(df['lists'].to_list())
    for feature in FEATURE_NAMES[4:]:
        df_cluster[feature] = df[feature]
    df_cluster['other_ASN'] = df['other_ASN']

    print(df_cluster)

    ct = ColumnTransformer([
        ('list_num_scaler', StandardScaler(), [x for x in range(0,1000)]),
        ('num_scaler', StandardScaler(), [x for x in range(1000,1013)]),
        ('cat_scaler', BinaryEncoder(), [x for x in range(1013,1014)])],
        remainder='passthrough')
    # ct = StandardScaler()

    df_cluster = ct.fit_transform(df_cluster)

    with open('encoders/ct.ColumnTransformer', 'wb') as numerical_file:
        pickle.dump(ct, numerical_file)
    
    k = 25
    clf = MiniBatchKMeans(n_clusters=k)
    X_dist = clf.fit_transform(df_cluster)

    with open('models/clf.MiniBatchKMeans', 'wb') as numerical_file:
        pickle.dump(clf, numerical_file)

    print(df_cluster)
    df_list = []
    for i in range(0,k):
        d = X_dist[:,i]
        ind = np.argsort(d)[::][:5]
        df_temp = df.loc[ind]
        df_temp['ip_other'] = df_temp['ip_other'].apply(lambda x : ntoa(x))
        df_temp['url_other'] = df_temp['ip_other'].apply(lambda x: resolveIP(x))
        num_members = (clf.labels_ == i).sum() / len(clf.labels_)
        dict = {'Cluster' : [i, i, i, i, i],
        'Membership' : [num_members,num_members,num_members, num_members, num_members],
        'Country' : df_temp['other_COUNTRY'],
        'URL' : df_temp['url_other'],
        'IP' : df_temp['ip_other']}
        df_list.append(pd.DataFrame(dict))
    df_centroids = pd.concat(df_list)
    print(df_centroids)

    print(df_centroids['IP'].value_counts(sort=True))
    print(df_centroids['Membership'])

    to_storage(df_centroids, 'centroid')

    return df_centroids, ct

def analyzeClusters():
    df2 = from_storage('centroid.parquet.gzip')
    df2['URL'] = df2.apply(resolveUnknownIP, axis=1)

    for i in range(0,50):
        print('CLUSTER {0}'.format(i))
        print(df2.loc[df2['Cluster'] == i][['URL','Country']])
        #print(df2.loc[df2['Cluster'] == i][['IP','URL']])
        print('______________________')

def classifyFlows():

    with open('models/clf.MiniBatchKMeans', 'rb') as numerical_file:
        clf = pickle.load(numerical_file)

    with open('encoders/ct.ColumnTransformer', 'rb') as numerical_file:
        ct = pickle.load(numerical_file)

    number = 10
    df = readDataset(CLUST_DATAFRAME_PATH, number=number)

    df_cluster = pd.DataFrame(df['lists'].to_list())
    for feature in FEATURE_NAMES[4:]:
        df_cluster[feature] = df[feature]
    df_cluster['other_ASN'] = df['other_ASN']

    print(df_cluster.head())
    print(len(df_cluster.columns))

    print('predicting...')
    df_cluster = ct.transform(df_cluster)
    df_pred = clf.predict(df_cluster)

    #Create prediction labels

    df['label'] = df_pred
    df['IP'] = df['ip_other'].apply(lambda x : ntoa(x))
    df['URL'] = df['IP'].apply(lambda x: resolveIP(x))
    #df['URL'] = df.apply(resolveUnknownIP, axis=1)
    df.drop(['lists', 'ip_other'], axis=1, inplace=True)
    df['pred_behavior'] = df['label'].apply(lambda x: bDict[x])

    print(df[['URL', 'pred_behavior']])
    df.to_csv('result{0}.csv'.format(number))

def getClusterInfo():
    with open('models/clf.MiniBatchKMeans', 'rb') as numerical_file:
        clf = pickle.load(numerical_file)

    print(clf.labels_)
    members = [(i, (clf.labels_ == i).sum()) for i in range(0,50)]
    
    members_ = sorted(members, key=lambda x: x[1], reverse=True)
    print(members_)

def createTrainingDataset(datasetType):
    if datasetType == 'training':
        df = readDataset()
    else:
        df = readDataset(path=CLUST_DATAFRAME_PATH, number=5)

    df_cluster = expandDataset(df)

    with open('models/clf.MiniBatchKMeans', 'rb') as numerical_file:
        clf = pickle.load(numerical_file)

    with open('encoders/ct.ColumnTransformer', 'rb') as numerical_file:
        ct = pickle.load(numerical_file)

    print('predicting...')
    df_cluster = ct.transform(df_cluster)
    df_pred = clf.predict(df_cluster)

    df['label'] = [BEHAVIOR_TABLE[x] for x in df_pred]
    df.drop(['ip_other'], axis=1, inplace=True)

    if datasetType == 'training':
        to_storage(df, 'training')
    else:
        to_storage(df, 'val')

    return df

def classifyFutureFlows():
    print('reading files....')
    filename = 'training.parquet.gzip'
    df = from_storage(filename)

    print(df['label'].unique())

    filename = 'val.parquet.gzip'
    X_val = from_storage(filename)

    print(X_val['label'].unique())

    print('cleaning dataset...')
    df = clean_dataframe(df)
    X_val = clean_dataframe(X_val)


    print('splitting dataframe...')
    df, df_target = split_dataframe(df)
    X_val, y_val = split_dataframe(X_val)

    print(df)
    print(X_val)

    print('encoding dataset...')
    df = expandDataset(df)
    # df.fillna(value=0, inplace=True) #FOR RFC
    df, ct = encode_dataframe(df)

    print(df.shape)
    print(df.dtype)

    X_val = expandDataset(X_val)
    # X_val.fillna(value=0, inplace=True) #FOR RFC
    X_val, _ = encode_dataframe(X_val, encoder=ct)


    print('Training model...')
    print(df_target)
    df_target_matrix, target_encoder = dataset_to_matrix(df_target, target_encoder=None)
    y_val_matrix, _ = dataset_to_matrix(y_val, target_encoder=target_encoder)
    print(df_target_matrix)

    flow_type_classes = ['Cloud Compupting', 'Google Search', 'Media', 'Microsoft Products']

    #RFC
    # target_encoder = LabelEncoder().fit(df_target)
    # df_target_matrix = target_encoder.transform(df_target)
    # model = RandomForestClassifier().fit(df, df_target_matrix)
    # val_pred_matrix = model.predict(X_val)
    # y_val_matrix = target_encoder.transform(y_val)
    # val_cm = confusion_matrix(y_val_matrix, val_pred_matrix)
    # val_report = classification_report(y_val_matrix, val_pred_matrix, target_names=flow_type_classes, digits=4, output_dict=False)

    model = simple_ANN(df, df_target_matrix, X_val, y_val_matrix)
    val_pred_matrix = model.predict(X_val)
    print(val_pred_matrix)
    val_pred = matrix_to_dataset(val_pred_matrix, encoder=target_encoder)
    val_cm = confusion_matrix(y_val, val_pred)
    val_report = classification_report(y_val,val_pred, target_names=flow_type_classes, digits=4, output_dict=False)

    print(val_report)
    print(val_cm)

def resolveLabels(value):
    if search('aws', value) is not None:
        result = 'Cloud Computing'
    elif search('google', value) is not None:
        result = 'Google Search'
    # elif search('')

    return result



def classifyFutureFlowsCluster():
    filename = 'val_csv.csv'
    df = pd.read_csv(filename)
    df.drop(labels=['Unnamed: 20', 'Unnamed: 21'], axis=1, inplace=True)


    drop_columns = [0, 6, 7, 9, 10, 11, 13, 16, 17, 18, 21, 22, 23]
    df = df[~df.label.isin(drop_columns)]
    df = df[df.pred_behavior != 'None']
    df = df[df.actual_behavior != 'None']

    print(df.actual_behavior.unique())
    print(df.pred_behavior.unique())

    val_cm = confusion_matrix(df.actual_behavior, df.pred_behavior)
    val_report = classification_report(df.actual_behavior, df.pred_behavior, output_dict=False)

    print(val_cm)
    print(val_report)

########## MAIN FUNCTION ##########
if __name__ == '__main__':
    start_time = datetime.now()

    # createBehaviorTable()
    # analyzeClusters()
    # classifyFlows()
    
    # getClusterInfo()
    
    # createTrainingDataset('training')
    # createTrainingDataset('val')
    
    # classifyFutureFlows()

    classifyFutureFlowsCluster()

    diff = datetime.now() - start_time
    print('Total runtime: {0}'.format(diff))
