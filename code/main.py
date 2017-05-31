import time
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np
from multiprocessing import Pool

label = 'logerror'
features = ['airconditioningtypeid', 'architecturalstyletypeid','basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid']

def input_fn(data_set):
    feature_cols = {}
    for k in features[:]:
        feature_cols[k] = tf.constant(data_set[k].values.astype(np.float32))
    labels = tf.constant(data_set.logerror.values)
    return feature_cols, labels

def pred(regressor, sub, pre, filename):
    pre['logerror'] = np.nan
    y = regressor.predict(input_fn=lambda: input_fn(pre))
    prediction = list(itertools.islice(y, sub.shape[0]))
    for i in sub.columns[1:]:
        sub[i] = prediction
    sub.rename(columns={'parcelid': 'ParcelId'}, inplace = True)
    sub.to_csv(filename, index = None, float_format='%.4f')

pool = Pool(10)
labelname = "../data/train_2016.csv"
featuresname = "../data/properties_2016.csv"
subname = "../data/sample_submission.csv"

label = pd.read_csv(labelname)
featuresdf = pd.read_csv(featuresname)
sub = pd.read_csv(subname)
sub.rename(columns={'ParcelId': 'parcelid'}, inplace = True)
date = range(len(np.unique(label.transactiondate)))
#print np.unique(label.transactiondate)
label.transactiondate.replace(np.unique(label.transactiondate), date, inplace = True)

df = label.merge(featuresdf, on = 'parcelid', how = 'left')
pre = sub.merge(featuresdf, on = 'parcelid', how = 'left')
#featurelist = features.columns.tolist()
#removelist = ['hashottuborspa','propertycountylandusecode','propertyzoningdesc','taxdelinquencyflag','fireplaceflag']
#for i in removelist:
    #featurelist.remove(i)
#featurelist = ['airconditioningtypeid', 'architecturalstyletypeid']#, 'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid']
#print len(featurelist)
df.fillna(0.0,inplace = True)
pre.fillna(0.0,inplace = True)
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in features]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[len(feature_cols)+1, 10],
                                          #model_dir="zillow_model2",
                                          #optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.005,l1_regularization_strength=0.001),
                                          )

start = time.time()
seperatetime = 10
regressor.fit(input_fn=lambda: input_fn(df[df.transactiondate < seperatetime]), steps=5000)

ev = regressor.evaluate(input_fn=lambda: input_fn(df[df.transactiondate >= seperatetime]), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}\nused time:{1}".format(loss_score,time.time()-start))

#pred(regressor, sub,pre, 'test.csv')
