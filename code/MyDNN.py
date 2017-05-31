import time
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
from multiprocessing import Pool
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

label = 'logerror'
features = ['airconditioningtypeid', 'architecturalstyletypeid','basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid']

def convertdataset(df):
    Dataset = collections.namedtuple('Dataset',['data','target'])
    return Dataset(data=np.array(df[features]),target=np.array(df[label]))

def preparedataframe():
    labelname = "../data/train_2016.csv"
    featuresname = "../data/properties_2016.csv"
    subname = "../data/sample_submission.csv"

    labeldf = pd.read_csv(labelname)
    featuresdf = pd.read_csv(featuresname)
    sub = pd.read_csv(subname)
    sub.rename(columns={'ParcelId': 'parcelid'}, inplace = True)
    date = range(len(np.unique(labeldf.transactiondate)))
    labeldf.transactiondate.replace(np.unique(labeldf.transactiondate), date, inplace = True)

    df = labeldf.merge(featuresdf, on = 'parcelid', how = 'left')
    pre = sub.merge(featuresdf, on = 'parcelid', how = 'left')
    df.fillna(0.0,inplace = True)
    pre.fillna(0.0,inplace = True)
    pre[label] = np.nan
    return df, pre


def pred(regressor, sub, pre, filename):
    pre['logerror'] = np.nan
    y = regressor.predict(input_fn=lambda: input_fn(pre))
    prediction = list(itertools.islice(y, sub.shape[0]))
    for i in sub.columns[1:]:
        sub[i] = prediction
    sub.rename(columns={'parcelid': 'ParcelId'}, inplace = True)
    sub.to_csv(filename, index = None, float_format='%.4f')


def model_fn(features, targets, mode, params):
    first_hidden_layer = tf.contrib.layers.relu(features, 10)
    second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer,10)
    output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)
    predictions = tf.reshape(output_layer, [-1])
    prediction_dict = {'logerror': predictions}
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
    seperatetime = 10

    def input_train_fn():
        train = convertdataset(df[df.transactiondate < seperatetime])
        return tf.constant(train.data), tf.constant(train.target)

    Mynn.fit(input_fn=input_train_fn, steps=5000)

    def input_test_fn():
        test = convertdataset(df[df.transactiondate >= seperatetime])
        return tf.constant(test.data), tf.constant(test.target)

    ev = Mynn.evaluate(input_fn=input_train_fn)
    print("Mean Absolute Error: {}".format(ev['mae']))
    '''
    predictions = Mynn.predict(x=predic.data, as_iterable=True)
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i + 1, p["logloss"]))
    '''
    #pred(regressor, sub,pre, 'test.csv')

if __name__ == "__main__":
    main()
