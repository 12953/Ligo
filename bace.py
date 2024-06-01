import tensorflow as tf
import numpy as np
from dataset_scaffold_random import Inference_Dataset
from model import PredictModel_test
from rdkit import Chem
import os

medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4,
          'd_model': 256, 'path': 'medium_weights', 'addH': True}

arch = medium  # small 3 4 128   medium: 6  6  256     large:  12 8 516
trained_epoch = 20
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']
addH = arch['addH']

dff = d_model * 2
vocab_size = 18
dropout_rate = 0.1

seed = 1
np.random.seed(seed=seed)
tf.random.set_seed(seed=seed)

os.environ['TF_DETERMINISTIC_OPS'] = '1'

'''
def predict(smi):
   
   # mol_list = []
  #  for smiles in smi:
	#    mol = Chem.MolFromSmiles(smiles)
   #     mol_list.append(mol)
   # img = Draw.MolsToGridImage(mol_list)


    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return -1
    ans = []
    y_preds = []
    res = []
    ligo=[]
    
        #    break
    #if 'file_path' not in dir():
    #    file_path = 'classification_weights/O00141_1.h5'

    for i in [smi]:
        x = [i]
      
    #if smi is not None:
    # 读取文件内容或者保存到特定路径
       # file_contents = smi.read()
       # with open("file.csv", "wb") as smi:
       #     smi.write(file_contents)   
    inference_dataset = Inference_Dataset(x,addH=addH).get_data()

    x, adjoin_matrix, smiles, atom_list  = next(
        iter(inference_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)

    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel_test(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                                dense_dropout=0.5)
    pred = model(x, mask=mask, training=False, adjoin_matrix=adjoin_matrix)
    for f_name in os.listdir('classification_weights'):
    # if f_name.startswith(id):
         
        file_path = f'classification_weights/{f_name}'
        model.load_weights(file_path)
        x, atts, xs = model(x, mask=mask, training=False,
                        adjoin_matrix=adjoin_matrix)
        y_preds.append(x)
    y_preds = tf.sigmoid(y_preds) 
    y_preds = tf.reshape(y_preds, (-1,))
    for i in y_preds.numpy():
        res.append(i)
    ligo.append([f_name,res[0]])
#return res[0]
    return ligo
'''
def predict(id, smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return -1
    ans = []
    y_preds = []
    res = []

    for f_name in os.listdir('classification_weights'):
        if f_name.startswith(id):
            file_path = f'classification_weights/{f_name}'
            break
    if 'file_path' not in dir():
        file_path = 'classification_weights/O00141_1.h5'

    for i in [smi]:
        x = [i]
        inference_dataset = Inference_Dataset(x, addH=addH).get_data()

        x, adjoin_matrix, smiles, atom_list = next(
            iter(inference_dataset.take(1)))
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)

        mask = seq[:, tf.newaxis, tf.newaxis, :]
        model = PredictModel_test(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                                  dense_dropout=0.5)
        pred = model(x, mask=mask, training=False, adjoin_matrix=adjoin_matrix)
        model.load_weights(file_path)

        x, atts, xs = model(x, mask=mask, training=False,
                            adjoin_matrix=adjoin_matrix)
        y_preds.append(x)
    y_preds = tf.sigmoid(y_preds)
    y_preds = tf.reshape(y_preds, (-1,))
    for i in y_preds.numpy():
        res.append(i)
    return res[0]
