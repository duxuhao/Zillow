import time
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
from multiprocessing import Pool
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

label = 'logerror'
labelname = "../data/train_2016.csv"
featuresname = "../data/properties_2016.csv"
subname = "../data/sample_submission.csv"
features = ['airconditioningtypeid', 
            'architecturalstyletypeid',
            'basementsqft', 
            'bathroomcnt', 
            'bedroomcnt', 
            'buildingclasstypeid',  
            'buildingqualitytypeid',
            'assessmentyear',
            'lotsizesquarefeet',
            #'finishedsquarefeet12',
            'finishedsquarefeet15',
            'fullbathcnt',
            'numberofstories',
            #'roomcnt',
            #'taxvaluedollarcnt',
            #'taxdelinquencyyear',
            #'taxdelinquencyflag',
            #'regionidcounty',
            'regionidcity',
            #'regionidzip',
            #'regionidneighborhood',
            ]

OneHotFeature = [[['regionidcity'], 'region']]

def convertdataset(df):
    if 'transactiondate' in df.columns:
        df.drop('transactiondate', inplace = True, axis=1)
    print df.shape
    Dataset = collections.namedtuple('Dataset',['data','target'])
    return Dataset(data=np.array(df.ix[:,df.columns != label]),target=np.array(df[label]))

def preparedataframe():
    labeldf = pd.read_csv(labelname)
    featuresdf = pd.read_csv(featuresname)
    featuresdf = featuresdf[features + ['parcelid']]
    sub = pd.read_csv(subname)
    sub.rename(columns={'ParcelId': 'parcelid'}, inplace = True)
    date = range(len(np.unique(labeldf.transactiondate)))
    labeldf.transactiondate.replace(np.unique(labeldf.transactiondate), date, inplace = True)
    '''
    for OHF in OneHotFeature[:1]:
        temp = featuresdf[OHF[0]].drop_duplicates()
        print temp.shape
        for onh in range(temp.shape[0]):
            temp['{}_{}'.format(OHF[1],onh)] = 0
            temp.loc[onh, '{}_{}'.format(OHF[1],onh)] = 1    
        featuresdf = featuresdf.merge(temp, on = OHF[0], how = 'left')        
    '''
    df = labeldf.merge(featuresdf, on = 'parcelid', how = 'left')
    pre = sub[['parcelid']].merge(featuresdf, on = 'parcelid', how = 'left')
    df.fillna(0.0,inplace = True)
    pre.fillna(0.0,inplace = True)
    pre[label] = np.nan
    return df, pre


def pred(regressor, pre, filename):
    sub = pd.read_csv(subname)
    pre[label] = np.nan
    def input_predict_fn():
        newpre = convertdataset(pre)
        return tf.constant(newpre.data), tf.constant(newpre.target)

    y = regressor.predict(input_fn=input_predict_fn, as_iterable=True)
    prediction = [i[label] for i in y]
    for i in sub.columns[1:]:
        sub[i] = prediction
    sub.rename(columns={'parcelid': 'ParcelId'}, inplace = True)
    sub.to_csv(filename, index = None, float_format='%.4f')


def model_fn(features, targets, mode, params):
    first_hidden_layer = tf.contrib.layers.relu(features, 16)
    second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer,10)
    #third_hidden_layer = tf.contrib.layers.relu(second_hidden_layer,5)
    output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

    predictions = tf.reshape(output_layer, [-1])
    prediction_dict = {label: predictions}

    loss = tf.losses.absolute_difference(targets, predictions)

    eval_metric_ops = {'mae': tf.metrics.mean_absolute_error(tf.cast(targets, tf.float64), predictions)}

    train_op = tf.contrib.layers.optimize_loss(
                                              loss=loss,
                                              global_step=tf.contrib.framework.get_global_step(),
                                              learning_rate=params['learning_rate'],
                                              optimizer='SGD')

    return model_fn_lib.ModelFnOps(
                                  mode=mode,
                                  predictions=prediction_dict,
                                  loss=loss,
                                  train_op=train_op,
                                  eval_metric_ops=eval_metric_ops)


def main():
    df, predic = preparedataframe()
    LEARNING_RATE = 0.1
    model_params = {"learning_rate": LEARNING_RATE}
    Mynn = tf.contrib.learn.Estimator(model_fn = model_fn, params=model_params)
    traintime = 300
    testtime = 300
    def input_train_fn():
        train = convertdataset(df[df.transactiondate < traintime])
        return tf.constant(train.data), tf.constant(train.target)
    start = time.time()
    Mynn.fit(input_fn=input_train_fn, steps=5000)

    def input_test_fn():
        test = convertdataset(df[df.transactiondate >= testtime])
        return tf.constant(test.data), tf.constant(test.target)

    #ev = Mynn.evaluate(input_fn=input_train_fn)
    #print("Mean Absolute Error: {}".format(ev['mae']))
    ev = Mynn.predict(input_fn=input_test_fn, as_iterable=True)
    evalvalue = [i[label] for i in ev]
    
    ev = np.mean(np.abs(np.array(evalvalue) - np.array(df[df.transactiondate >= testtime][label])))
    print(ev)
    pred(Mynn, predic, 'test1010_morefea.csv')
    print('used time: {} s'.format(time.time()-start))
if __name__ == "__main__":
    main()
