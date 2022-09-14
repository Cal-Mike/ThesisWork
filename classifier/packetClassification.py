# Multi-Dimensional Profiling of Cyber Threats in Large-Scale Networks code
# Created by Michael Calnan
# Last edited: 14 SEP 2022

########## IMPORTS ##########

from gc import callbacks
import gc
from pickletools import optimize
from turtle import shape
from xml.sax.handler import all_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from category_encoders.binary import BinaryEncoder
from keras.utils import np_utils
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from os import listdir
########## GLOBAL VARIABLES ##########

NUMERICAL_FEATURES = ['ip_len','src_LAT','src_LONG','dst_LAT','dst_LONG']
CATEGORICAL_FEATURES =['port_src','port_dst','src_ASN','dst_ASN','src_COUNTRY','dst_COUNTRY', 'ip_protocol']
DROP_FEATURES = ['timestamp'] #Features to drop from each dataset

TRAIN_DATASET_PATH = '/home/michael.calnan/calnan_thesis/classifier/datasets/training/'
VAL_DATASET_PATH = '/home/michael.calnan/calnan_thesis/classifier/datasets/validation/'
MODEL_PATH = '/home/michael.calnan/calnan_thesis/classifier/models/'
FLOW_DICT = {0: 'OUTGOING_BLOCKED', 1: 'INCOMING_BLOCKED', 2: 'OUTGOING_ALLOWED', 3: 'INCOMING_ALLOWED', 4: 'UNIDENTIFIED'}
FLOW_DICT_REPLACE = {1: 3, #Incoming -> Incoming Allowed
    2: 2, #Outgoing -> Outgoing Allowed
    3: 0, #Inside Blocked -> Outgoing Blocked
    4: 4, #Inside-not-allowed -> Unidentified
    5: 3, #Inside-dropped-inbound -> Incoming Allowed
    6: 2, #Inside-dropped-outbound -> Outgoing Allowed
    7: 1, #Outside-blocked -> Incoming Blocked
    8: 4, #Outside-not-allowed -> Unidentified
    9: 3, #Outside-dropped-inbound -> Incoming Allowed
    10: 2} #Outside-dropped-outbound -> Outgoing Allowed

TRAINING_SIZE = None            #For RUS, size of each class. Use -1 for maximum value
TRAINING_SAMPLE_SIZE = 2000000  #For Deep Learning. Max number of samples before memory errors
#TRAINING_SAMPLE_SIZE = 53000   #For RFC. Max number of samples before memory errors
MODEL_NAME = 'cnn'              #name of model: either 'rfc','ann', or 'cnn'
MODEL_TYPE = 'Transfer'
BATCH_SIZE = 512

#Training filenames
FILE_TRAIN_1 = 'dataset_2022-07-14_13'
FILE_TRAIN_2 = 'dataset_2022-07-14_14_00'
FILE_TRAIN_3 = 'dataset_2022-07-14_14_15'
FILE_TRAIN_4 = 'dataset_2022-07-14_14_30'
FILE_TRAIN_5 = 'dataset_2022-07-14_14_45'
FILE_TRAIN_6 = 'dataset_2022-07-14_15'

#Validation filenames
FILE_VAL_15 = 'dataset_2022-07-15_14'
FILE_VAL_16 = 'dataset_2022-07-16_14'
FILE_VAL_17 = 'dataset_2022-07-17_14'
FILE_VAL_18 = 'dataset_2022-07-18_14'
FILE_VAL_19 = 'dataset_2022-07-19_14'
FILE_VAL_20 = 'dataset_2022-07-20_10'

########## DEEP LEARNING MODELS ##########

def cnn_1d_classifier(num_features):
    '''Create 1D Convolutional Neural Network.
    '''
    # X_train = np.array(X_train)
    # X_val = np.array(X_val)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,input_shape=[num_features, 1]),
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
    return model

def simple_ANN(num_features):
    '''Create Multi-Layer Perceptron.
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(num_features,)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    print(model.summary())
    return model

def deep_fit(model, train_dataset, X_val_f, y_val_f, total_samples, classweight=None, train_dataset2=None):
    '''Trains Deep Learning Model on training data.
    '''
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

    #Create piecewise learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return 0.001
        elif epoch < 20:
            return 1e-5
        else:
            return 1e-5
    lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    #Train model
    if train_dataset2 is not None:
        model.fit(
            x=train_dataset2,
            #steps_per_epoch= total_samples // BATCH_SIZE,
            steps_per_epoch= 13178, #For RUS
            #steps_per_epoch= 60849, #For everything else
            validation_data=(X_val_f, y_val_f),
            verbose=1,
            epochs=50, 
            callbacks=[es, lrs]#,
            #class_weight=classweight
        )
    
    history = model.fit(
        x=train_dataset,
        #steps_per_epoch= total_samples // BATCH_SIZE,
        #steps_per_epoch= 13178, #For RUS
        steps_per_epoch= 60849, #For everything else
        validation_data=(X_val_f, y_val_f),
        verbose=1,
        epochs=50, 
        callbacks=[es, lrs],
        class_weight=classweight
    )

    return model

########## BASELINE RANDOM FOREST TREE ##########

def simple_rft_classifier(X_train, y_train):
    '''Create Random Forest Classifier
    '''
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print('feature importance') #print sorted gini feature importance
    feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns.tolist()).sort_values(ascending=False)
    print(feature_imp)

    return clf

########## METRICS ANALYSIS ###########

def plot_days(ReportList, filename='dayplots') -> None:
    '''Plots performance over several days of data.

    ReportList (List<dict>): List containing sklearn classification reports in dictionary format
    '''
    print('...plotting multi-day metrics')
    outBlockedDict = {'f1-score': []}
    inBlockedDict = {'f1-score': []}
    outAllowedDict = {'f1-score': []}
    inAllowedDict = {'f1-score': []}
    accuracy = {'accuracy': []}
    TOTAL_DAYS = len(ReportList)

    for report in ReportList:
        outBlockedDict['f1-score'].append(report['OUTGOING_BLOCKED']['f1-score'])
        inBlockedDict['f1-score'].append(report['INCOMING_BLOCKED']['f1-score'])
        outAllowedDict['f1-score'].append(report['OUTGOING_ALLOWED']['f1-score'])
        inAllowedDict['f1-score'].append(report['INCOMING_ALLOWED']['f1-score'])
        accuracy['accuracy'].append(report['accuracy'])
    print(inBlockedDict)
    
    #CREATE GRAPH FOR EACH LABEL'S F1-SCORE
    plt.clf()
    fig = plt.figure()
    plt.xlabel('Days Since Training')
    plt.ylabel('F1-Score')
    plt.title('Duration Effects on Model F1-Score')
    plt.plot(range(0,TOTAL_DAYS), outBlockedDict['f1-score'], color='red', label='Out Blocked f1-score')
    plt.plot(range(0,TOTAL_DAYS), inBlockedDict['f1-score'], color='red', label='In Blocked f1-Score', linestyle='dashed')
    plt.plot(range(0,TOTAL_DAYS), outAllowedDict['f1-score'], color='blue', label='Out Allowed f1-score')
    plt.plot(range(0,TOTAL_DAYS), inAllowedDict['f1-score'], color='blue', label='In Allowed f1-Score', linestyle='dashed')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(filename+'_f1.png')

    #CREATE GRAPH FOR TOTAL ACCURACY
    plt.clf()
    fig = plt.figure()
    plt.xlabel('Days Since Training')
    plt.ylabel('Accuracy')
    plt.title('Duration Effects on Model Accuracy')
    plt.plot(range(0,TOTAL_DAYS), accuracy['accuracy'], color='black')
    plt.grid()
    plt.show()
    plt.savefig(filename+'_accuracy.png')

def plot_prcurve(y_predictions, y_true, filename) -> None:
    '''Plots the Precision-Recall Curve in Matlibplot, saves under filename.png

    y_precitions (nparray): Size [x,4] of prediction values
    y_true (nparray): Size [x,4] of truth values
    filename (String): Name to save file to in form of 'filename.png'
    '''
    print('...plotting precision-recall curve')
    outBlockedDict = {'true' : y_true[:,0], 'pred': y_predictions[:,0]}
    inBlockedDict = {'true' : y_true[:,1], 'pred': y_predictions[:,1]}
    outAllowedDict = {'true' : y_true[:,2], 'pred': y_predictions[:,2]}
    inAllowedDict = {'true' : y_true[:,3], 'pred': y_predictions[:,3]}

    outBlockPR = precision_recall_curve(outBlockedDict['true'], outBlockedDict['pred'])
    inBlockPR = precision_recall_curve(inBlockedDict['true'], inBlockedDict['pred'])
    outAllowedPR = precision_recall_curve(outAllowedDict['true'], outAllowedDict['pred'])
    inAllowedPR = precision_recall_curve(inAllowedDict['true'], inAllowedDict['pred'])

    plt.clf()
    fig = plt.figure()

    PrecisionRecallDisplay(outBlockPR[0], outBlockPR[1]).plot(ax=plt.gca(), name='Precision-recall for Outgoing Blocked', color='red')
    PrecisionRecallDisplay(inBlockPR[0], inBlockPR[1]).plot(ax=plt.gca(), name='Precision-recall for Incoming Blocked', color='red', linestyle='dashed')
    PrecisionRecallDisplay(outAllowedPR[0], outAllowedPR[1]).plot(ax=plt.gca(), name='Precision-recall for Outgoing Allowed', color='blue')
    PrecisionRecallDisplay(inAllowedPR[0], inAllowedPR[1]).plot(ax=plt.gca(), name='Precision-recall for Incoming Allowed', color='blue', linestyle='dashed')

    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.show()
    plt.savefig(filename + '.png')

########## DATA PRE-PROCESSING ##########

def from_storage(fileno, set_type):
    '''Read Parquet file from local dataset directory.
    '''
    if set_type == 'validation':
        file_data = pd.read_parquet('{0}{1}.parquet.gzip'.format(VAL_DATASET_PATH, fileno))
        file_len = file_data.shape[0]    
    else: #for training
        file_data = pd.read_parquet('{0}{1}.parquet.gzip'.format(TRAIN_DATASET_PATH, fileno))
        file_len = file_data.shape[0]
    return file_data, file_len

def clean_dataframe(df):
    '''Remove erronous columns from dataset.
    '''
    #Change 9 labels to 4 classes, remove other classes
    df['packet_label'] = df['packet_label'].replace(FLOW_DICT_REPLACE)
    df = df[df.packet_label != 4]

    #drop ip addresses and index columns
    if 'ip_src' in df.columns and 'ip_dst' in df.columns:
        df = df.drop(labels=['ip_src','ip_dst'],axis=1,inplace=False)
    if 'index' in df.columns:
        df = df.drop(labels=['index'], axis=1,inplace=False)

    df = df.drop(labels=DROP_FEATURES, axis=1, inplace=False)

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

def split_dataframe(df, size=None):
    '''Create target and feature vector arrays.
    '''
    #If random-unsampling is used
    if size is not None:
        vc = df['packet_label'].value_counts(ascending=False).min()
        if size > 0 and size < vc: #If -1 or too large, set to maximum potential sampling size
            vc = size
        df = df.groupby('packet_label').sample(n=vc) #Get equal samples from each class
    
    df_target = df['packet_label']
    df.pop('packet_label')
    return df, df_target

def encode_dataframe(df, ct_scaler=None):
    '''Encode entire dataset to values between 0 and 1.
    '''
    if ct_scaler is None:
        ct_scaler = ColumnTransformer([
            ('numerical features', StandardScaler(),NUMERICAL_FEATURES),
            ('categorical features', BinaryEncoder(cols=CATEGORICAL_FEATURES), CATEGORICAL_FEATURES)
        ], remainder='passthrough').fit(df)
    df_encoded = ct_scaler.transform(df)

    return df_encoded, ct_scaler

def generate_batches(files, batch_size, feature_number, dir, ct_scaler,target_encoder):
    counter = 0
    while True:
        counter = counter % len(files)
        print('\n{0} of {1}'.format(counter+1, len(files)))
        file = files[counter]
        print(file)
        df = pd.read_parquet(dir + file)

        #print(df.dtypes)

        #PREPROCESSING HERE
        df = clean_dataframe(df)
        df, output = split_dataframe(df, size=TRAINING_SIZE)
        
        input, _ = encode_dataframe(df, ct_scaler)

        if MODEL_NAME == 'cnn':
            input = tf.reshape(input, (input.shape[0], feature_number, 1))
        
        print(input.shape)
        output, _ = dataset_to_matrix(output)

        for idx in range(0, input.shape[0], batch_size):
            input_local = input[idx:(idx+batch_size)]
            output_local = output[idx:(idx+batch_size)]
            yield input_local, output_local
        counter += 1

def generate_batches_RUS(files, batch_size, feature_number, dir, ct_scaler,target_encoder):
    counter = 0
    while True:
        counter = counter % len(files)
        print('\n{0} of {1}'.format(counter+1, len(files)))
        file = files[counter]
        print(file)
        df = pd.read_parquet(dir + file)

        #print(df.dtypes)

        #PREPROCESSING HERE
        df = clean_dataframe(df)
        df, output = split_dataframe(df, size=-1)
        
        input, _ = encode_dataframe(df, ct_scaler)

        if MODEL_NAME == 'cnn':
            input = tf.reshape(input, (input.shape[0], feature_number, 1))
        
        print(input.shape)
        output, _ = dataset_to_matrix(output)

        for idx in range(0, input.shape[0], batch_size):
            input_local = input[idx:(idx+batch_size)]
            output_local = output[idx:(idx+batch_size)]
            yield input_local, output_local
        counter += 1

def rfc_model_analysis(df, val_list):
    print('cleaning data...')
    
    #training set
    df_ = clean_dataframe(df)
    df_.fillna(value='0', inplace=True)

    country_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(df_[['src_COUNTRY', 'dst_COUNTRY']])
    df_[['src_COUNTRY_enc', 'dst_COUNTRY_enc']] = country_encoder.transform(df_[['src_COUNTRY', 'dst_COUNTRY']])
    df_ = df_.drop(labels=['src_COUNTRY', 'dst_COUNTRY'], axis=1, inplace=False)

    print(df_)
    X, y = split_dataframe(df_, size=None)

    print('enoding data...')
    #Training and validation datsets plus scalers
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    print('Training shape: {0}'.format(X_train.shape))
    
    # with open('encoders/num_scaler.MinMaxScaler', 'wb') as numerical_file:
    #     pickle.dump(num_scaler, numerical_file)
    # with open('encoders/cat_scaler.BinaryEncoder', 'wb') as categorical_file:
    #     pickle.dump(cat_scaler, categorical_file)
    
    print('***training model***')
    model = simple_rft_classifier(X_train, y_train)
    
    print('**model analysis')
    flow_type_classes = list(FLOW_DICT.values())[:-1]
    reportList = []

    train_pred = model.predict(X_train)
    train_cm = confusion_matrix(y_train, train_pred)
    train_report = classification_report(y_train,train_pred, target_names=flow_type_classes, digits=4, output_dict=True)
    print(train_report)
    print(train_cm)
    reportList.append(train_report)

    current_day = 15
    for df_val in val_list:
        print('*****************\nJULY {0}\n*****************'.format(current_day))
        df_val = clean_dataframe(df_val)
        
        df_val[['src_COUNTRY_enc', 'dst_COUNTRY_enc']] = country_encoder.transform(df_val[['src_COUNTRY', 'dst_COUNTRY']])
        df_val = df_val.drop(labels=['src_COUNTRY', 'dst_COUNTRY'], axis=1, inplace=False)

        X_val2, y_val2 = split_dataframe(df_val, size=None)

        val2_pred = model.predict(X_val2)
        val2_cm = confusion_matrix(y_val2, val2_pred)
        val2_report = classification_report(y_val2,val2_pred, target_names=flow_type_classes, digits=4,output_dict=True)
        print(val2_report)
        print(val2_cm)
        reportList.append(val2_report)
        current_day += 1


        del df_val

    plot_days(reportList, filename='dayplots_rfc')

def dl_model_analysis(df, val_list, total_samples):
    print('cleaning data...')
    
    df_ = clean_dataframe(df)
    print(df_)
    X, y = split_dataframe(df_, size=None)

    print('enoding data...')
    #Training and validation datsets plus scalers
    X_, ct_scaler = encode_dataframe(X, ct_scaler=None)
    X_train, X_val, y_train, y_val = train_test_split(X_, y, test_size=0.2)
    print('Training shape: {0}'.format(X_train.shape))

    # with open('encoders/ct_scaler.ColumnTransformer', 'wb') as numerical_file:
    #     pickle.dump(ct_scaler, numerical_file)

    #encode training and validation labels (comment out for random forest)
    y_train_matrix, target_encoder = dataset_to_matrix(y_train)
    y_val_matrix, _ = dataset_to_matrix(y_val, target_encoder=target_encoder)

    # with open('encoders/target_encoder.LabelEncoder', 'wb') as label_encoder_file:
    #     pickle.dump(target_encoder, label_encoder_file)

    #Create class weights
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
    print(classweight)

    if MODEL_NAME == 'cnn':
        val_batch, num_features = X_val.shape
        X_val = tf.reshape(X_val, (val_batch, num_features, 1))
        
        model = cnn_1d_classifier(num_features)
    else:
        train_batch, num_features = X_train.shape
        model = simple_ANN(num_features)

    print('freeing memory...')
    del y_train_matrix
    del y_train
    del X_train
    del X
    del X_
    gc.collect()

    print('creating training dataset...')
    train_filenames = listdir(TRAIN_DATASET_PATH)
    if MODEL_NAME == 'ann':
        train_dataset = tf.data.Dataset.from_generator(
            generator=lambda: generate_batches(
                                            files=train_filenames, 
                                            batch_size=BATCH_SIZE,
                                            feature_number=num_features, 
                                            dir=TRAIN_DATASET_PATH,
                                            ct_scaler=ct_scaler,
                                            target_encoder=target_encoder),
                # output_signature=(
                #     tf.TensorSpec(shape=(None, num_features), dtype=tf.float32),
                #     tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                output_types=(tf.float32, tf.float32),
                output_shapes=((None, num_features), (None, 4))
            )

        train_dataset2 = tf.data.Dataset.from_generator(
            generator=lambda: generate_batches_RUS(
                                            files=train_filenames, 
                                            batch_size=BATCH_SIZE,
                                            feature_number=num_features, 
                                            dir=TRAIN_DATASET_PATH,
                                            ct_scaler=ct_scaler,
                                            target_encoder=target_encoder),
                # output_signature=(
                #     tf.TensorSpec(shape=(None, num_features), dtype=tf.float32),
                #     tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                output_types=(tf.float32, tf.float32),
                output_shapes=((None, num_features), (None, 4))
            )
    else:
        train_dataset = tf.data.Dataset.from_generator(
            generator=lambda: generate_batches(
                                            files=train_filenames, 
                                            batch_size=BATCH_SIZE,
                                            feature_number=num_features, 
                                            dir=TRAIN_DATASET_PATH,
                                            ct_scaler=ct_scaler,
                                            target_encoder=target_encoder),
                # output_signature=(
                #     tf.TensorSpec(shape=(None, feature_number, 1), dtype=tf.float32),
                #     tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                output_types=(tf.float32, tf.float32),
                output_shapes=((None, num_features, 1), (None, 4))
            )

        train_dataset2 = tf.data.Dataset.from_generator(
            generator=lambda: generate_batches_RUS(
                                            files=train_filenames, 
                                            batch_size=BATCH_SIZE,
                                            feature_number=num_features, 
                                            dir=TRAIN_DATASET_PATH,
                                            ct_scaler=ct_scaler,
                                            target_encoder=target_encoder),
                # output_signature=(
                #     tf.TensorSpec(shape=(None, feature_number, 1), dtype=tf.float32),
                #     tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                output_types=(tf.float32, tf.float32),
                output_shapes=((None, num_features, 1), (None, 4))
            )


    
    print('***training model***')
    model = deep_fit(model, train_dataset, X_val, y_val_matrix, total_samples, classweight=classweight, train_dataset2=train_dataset2)
    
    print('Saving model...')
    model.save('{0}{1}{2}'.format(MODEL_PATH, MODEL_NAME,MODEL_TYPE))
    #model = tf.load

    print('**model analysis')
    flow_type_classes = list(FLOW_DICT.values())[:-1]
    reportList = []

    print('training set metrics...')
    train_pred_matrix = model.predict(X_val)
    train_pred = matrix_to_dataset(train_pred_matrix, encoder=target_encoder)
    train_cm = confusion_matrix(y_val, train_pred)
    train_report = classification_report(y_val,train_pred, target_names=flow_type_classes, digits=4, output_dict=True)
    print(train_report)
    print(train_cm)
    reportList.append(train_report)

    current_day = 15
    for df_val in val_list:
        print('*****************\nJULY {0}\n*****************'.format(current_day))
        df_val_ = clean_dataframe(df_val)
        X_val2, y_val2 = split_dataframe(df_val_, size=None)
        X_val2, _ = encode_dataframe(X_val2, ct_scaler=ct_scaler)
        y_val2_matrix, _ = dataset_to_matrix(y_val2, target_encoder=target_encoder)
        print('shape: {0}'.format(X_val2.shape))

        if MODEL_NAME == 'cnn':
            val2_batch, n_features = X_val2.shape
            X_val2 = tf.reshape(X_val2, (val2_batch, n_features, 1))
        

        plot_prcurve(val2_pred_matrix, y_val2_matrix, filename='{0}_{1}_july_{2}_PRCurve'.format(MODEL_NAME, MODEL_TYPE, current_day))
        reportList.append(val2_report)
        
        del df_val
        del df_val_
        del X_val2
        del y_val2
        gc.collect()
        
        current_day += 1

    plot_days(reportList, filename='dayplots_{0}_{1}'.format(MODEL_NAME, MODEL_TYPE))

########## MAIN FUNCTION ##########

if __name__ == '__main__':
    start_time = datetime.now()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.test.is_built_with_cuda)
    print(tf.config.get_visible_devices())

    print('***preparing data***')

    print('reading from storage...')
    df, df_len1 = from_storage(FILE_TRAIN_1, 'train')
    df_train2, df_len2 = from_storage(FILE_TRAIN_2, 'train')
    df_train3, df_len3 = from_storage(FILE_TRAIN_3, 'train')
    df_train4, df_len4 = from_storage(FILE_TRAIN_4, 'train')
    df_train5, df_len5 = from_storage(FILE_TRAIN_5, 'train')
    df_train6, df_len6 = from_storage(FILE_TRAIN_6, 'train')
    df = pd.concat([df.sample(n=TRAINING_SAMPLE_SIZE),
        df_train2.sample(n=TRAINING_SAMPLE_SIZE), 
        df_train3.sample(n=TRAINING_SAMPLE_SIZE), 
        df_train4.sample(n=TRAINING_SAMPLE_SIZE), 
        df_train5.sample(n=TRAINING_SAMPLE_SIZE), 
        df_train6.sample(n=TRAINING_SAMPLE_SIZE)])
    df.reset_index(inplace=True)
    total_samples = df_len1 + df_len2 + df_len3 + df_len4 + df_len5 + df_len6

    print('Flow types for training data')
    print(df['packet_label'].value_counts(dropna=False))

    df_val_15, _ = from_storage(FILE_VAL_15, 'validation')
    df_val_16, _ = from_storage(FILE_VAL_16, 'validation')
    df_val_17, _ = from_storage(FILE_VAL_17, 'validation')
    df_val_18, _ = from_storage(FILE_VAL_18, 'validation')
    df_val_19, _ = from_storage(FILE_VAL_19, 'validation')
    df_val_20, _ = from_storage(FILE_VAL_20, 'validation')

    if MODEL_NAME == 'rfc':
        rfc_model_analysis(df, [df_val_15, df_val_16, df_val_17, df_val_18, df_val_19, df_val_20])
    else:
        dl_model_analysis(df, [df_val_15, df_val_16, df_val_17, df_val_18, df_val_19, df_val_20], total_samples)

    diff = datetime.now() - start_time
    print('Total runtime: {0}'.format(diff))