import tensorflow as tf
import pandas as pd
import numpy as np


def input_fn(data_set, features, label):
    feature_cols = {}
    for k in features:
        #print data_set[k].values
        feature_cols[k] = tf.constant(data_set[k].values)
    #feature_cols = {k: tf.constant(data_set[k].values) for k in features}
    labels = tf.constant(data_set[label].values)
    return feature_cols, labels


labelname = '../data/train_2016.csv'
featuresname = "../data/properties_2016.csv"

label = pd.read_csv(labelname)
features = pd.read_csv(featuresname)
date = range(len(np.unique(label.transactiondate)))
label.transactiondate.replace(np.unique(label.transactiondate), date, inplace = True)

df = label.merge(features, on = 'parcelid', how = 'left')
featurelist = features.columns.tolist()
featurelist.remove('propertycountylandusecode')
featurelist.remove('propertyzoningdesc')
featurelist.remove('taxdelinquencyflag')
featurelist = ['airconditioningtypeid', 'architecturalstyletypeid', 'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid']
df.fillna(-100,inplace = True)

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in featurelist]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[len(feature_cols)+1, 10],
                                          model_dir="zillow_model")

regressor.fit(input_fn=lambda: input_fn(df[df.transactiondate < 250], featurelist, 'logerror'), steps=5000)
ev = regressor.evaluate(input_fn=lambda: input_fn(df[df.transactiondate >= 250], featurelist, 'logerror'), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))
