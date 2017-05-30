import time
import tensorflow as tf
import pandas as pd
import numpy as np
from multiprocessing import Pool

def input_fn(data_set, features, label='None'):
    feature_cols = {}
    for k in features[:]:
        feature_cols[k] = tf.constant(data_set[k].values)
    #feature_cols = {k: tf.constant(data_set[k].values) for k in features}
    if label == 'None':
        return feature_cols
    else:
        labels = tf.constant(data_set[label].values)
        return feature_cols, labels

pool = Pool(10)
labelname = "../data/train_2016.csv"
featuresname = "../data/properties_2016.csv"
subname = "../data/sample_submission.csv"

label = pd.read_csv(labelname)
features = pd.read_csv(featuresname)
sub = pd.read_csv(subname)
sub.rename(columns={'ParcelId': 'parcelid'}, inplace = True)
date = range(len(np.unique(label.transactiondate)))
#print np.unique(label.transactiondate)
label.transactiondate.replace(np.unique(label.transactiondate), date, inplace = True)

df = label.merge(features, on = 'parcelid', how = 'left')
pre = sub.merge(features, on = 'parcelid', how = 'left')
featurelist = features.columns.tolist()
removelist = ['hashottuborspa','propertycountylandusecode','propertyzoningdesc','taxdelinquencyflag','fireplaceflag']
for i in removelist:
    featurelist.remove(i)
featurelist = ['airconditioningtypeid', 'architecturalstyletypeid', 'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid']
print len(featurelist)
df.fillna(0.0,inplace = True)

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in featurelist]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[len(feature_cols)+1, 10],
                                          #model_dir="zillow_model2",
                                          #optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.005,l1_regularization_strength=0.001),
                                          )

start = time.time()
seperatetime = 250
regressor.fit(input_fn=lambda: input_fn(df[df.transactiondate < seperatetime], featurelist, 'logerror'), steps=5000)
ev = regressor.evaluate(input_fn=lambda: input_fn(df[df.transactiondate >= seperatetime], featurelist, 'logerror'), steps=1)
prediction = regressor.fit(input_fn=lambda: input_fn(pre, featurelist), step = 1)
for i in sub.columns[1:]:
    sub[i] = prediction
loss_score = ev["loss"]
print("Loss: {0:f}\nused time:{1}".format(loss_score,time.time()-start))
sub.to_csv('test.csv', index = None, float_format='%.4f')
