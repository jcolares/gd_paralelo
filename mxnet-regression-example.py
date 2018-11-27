import mxnet as mx
import numpy as np
import pandas as pd 
#import matplotlib as plt

# Fix the random seed
mx.random.seed(42)

import logging
logging.getLogger().setLevel(logging.DEBUG)

''' **** MXNET SAMPLE DATA ****
#Training data
train_data = np.random.uniform(0, 1, [100, 2])
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])
batch_size = 1

#Evaluation Data
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11,26,16])
'''

#Carregar dados 
df = pd.read_csv('diamond_prices.csv', header=0)
df.carat = (df.carat - df.carat.mean()) / (df.carat.max() - df.carat.min())
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

train_X = np.array(train[['carat']])
train_y = np.array(train[['price']])

test_X = np.array(test[['carat']])
test_y = np.array(test[['price']])

train_X_norm = train_X
test_X_norm = test_X

#train_X_norm = (train_X - train_X.mean()) / (train_X.max() - train_X.min())
#test_X_norm = (test_X - train_X.mean()) / (test_X.max() - test_X.min())

# Tamanho do batch (batch GD, minibatch GD)
batch_size = 1000

#Iterators - utilizados para gerar os lotes ou mini-lotes de dados de treinamento para o modelo
#train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, shuffle=True, label_name='lin_reg_label')
#eval_iter =  mx.io.NDArrayIter(eval_data,  eval_label,  batch_size, shuffle=False, label_name='lin_reg_label')
train_iter = mx.io.NDArrayIter(train_X_norm, train_y, batch_size, shuffle=True, label_name='lin_reg_label')
eval_iter =  mx.io.NDArrayIter(test_X_norm,  test_y,  batch_size, shuffle=False, label_name='lin_reg_label')

#Network 
X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

#Model
model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)

#Visualize network
netw = mx.viz.plot_network(symbol=lro, node_attrs={"shape":"oval","fixedsize":"false"})
netw.format = 'png'
netw.render('netw')

#Training
model.fit(train_iter, eval_iter,
        optimizer_params={'learning_rate':0.005, 'momentum': 0.9},
        num_epoch=100,
        eval_metric='mse',
        #batch_end_callback = mx.callback.LogValidationMetricsCallback)
        batch_end_callback = mx.callback.Speedometer(batch_size, 2))

#Evaluate
metric = mx.metric.MSE()
mse = model.score(eval_iter, metric)
print("Achieved {0:.6f} validation MSE".format(mse[0][1]))
#assert model.score(eval_iter, metric)[0][1] < 0.01001, "Achieved MSE (%f) is larger than expected (0.01001)" % model.score(eval_iter, metric)[0][1]            
print("Achieved MSE (%f) is larger than expected (0.01001)" % model.score(eval_iter, metric)[0][1] )           

# Predict
# model.predict(eval_iter).asnumpy()
