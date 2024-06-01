import re
from cProfile import label
from cgi import test
from tkinter import Label
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import smiles2adjoin
from rdkit import Chem
from random import Random
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
"""     

{'O': 5000757, 'C': 34130255, 'N': 5244317, 'F': 641901, 'H': 37237224, 'S': 648962, 
'Cl': 373453, 'P': 26195, 'Br': 76939, 'B': 2895, 'I': 9203, 'Si': 1990, 'Se': 1860, 
'Te': 104, 'As': 202, 'Al': 21, 'Zn': 6, 'Ca': 1, 'Ag': 3}

H C N O F S  Cl P Br B I Si Se
"""


# str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
#          'B': 10,'I': 11,'Si':12,'Se':13,'<unk>':14,'<mask>':15,'<global>':16}

# num2str =  {i:j for j,i in str2num.items()}

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}

num2str =  {i:j for j,i in str2num.items()}


def generate_scaffold(mol, include_chirality=False):
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(smiles, use_indices=False):
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param smiles: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, smi in enumerate(smiles):
        scaffold = generate_scaffold(smi)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smi)

    return scaffolds


def scaffold_split(pyg_dataset, sizes=(0.8, 0.1, 0.1), balanced=True, seed=1):

    assert sum(sizes) == 1

    # Split
    print('generating scaffold......')
    num = len(pyg_dataset)
    train_size, val_size, test_size = sizes[0] * num, sizes[1] * num, sizes[2] * num
    train_ids, val_ids, test_ids = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    smiles = 'smiles'
    scaffold_to_indices = scaffold_to_smiles(pyg_dataset[smiles], use_indices=True)

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train_ids) + len(index_set) <= train_size:
            train_ids += index_set
            train_scaffold_count += 1
        elif len(val_ids) + len(index_set) <= val_size:
            val_ids += index_set
            val_scaffold_count += 1
        else:
            test_ids += index_set
            test_scaffold_count += 1

    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                 f'train scaffolds = {train_scaffold_count:,} | '
                 f'val scaffolds = {val_scaffold_count:,} | '
                 f'test scaffolds = {test_scaffold_count:,}')

    print(f'Total smiles = {num:,} | '
                 f'train smiles = {len(train_ids):,} | '
                 f'val smiles = {len(val_ids):,} | '
                 f'test smiles = {len(test_ids):,}')

    assert len(train_ids) + len(val_ids) + len(test_ids) == len(pyg_dataset)

    return train_ids, val_ids, test_ids

class Graph_Bert_Dataset(object):  #  MG-BERT模型数据处理
    def __init__(self,path,smiles_field='Smiles',addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):

        data = self.df
        train_idx = []
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(256, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, smiles):   #  数值表示smiles
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)  # 得到 return atoms_list,adjoin_matrix
       
        atoms_list = ['<global>'] + atoms_list
        
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]  #  将原子转换为数字,字典中有 i，列表中添加i对应的值，如果字典中无i，添加str2num['<unk>']对应的值14
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)


        choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1  # 生成一个随机排序的列表，切片15%，至少有一个
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:  #  在15%的原子中
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']  #  随机选择0.8掩蔽
            elif rand < 0.9:  #  0.1替换为其它原子
                nums_list[i] = int(np.random.rand() * 14 + 1)
                              #  0.1保持不变
        x = np.array(nums_list).astype('int64')
        
        weight = weight.astype('float32')
        

        return x, adjoin_matrix, y, weight

    def tf_numerical_smiles(self, data):  #  转换类型和维度
        # x,adjoin_matrix,y,weight = tf.py_function(self.balanced_numerical_smiles,
        #                                           [data], [tf.int64, tf.float32 ,tf.int64,tf.float32])
        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
                                                     [tf.int64, tf.float32, tf.int64, tf.float32])

        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix, y, weight

class Graph_Classification_Dataset(object):  # 图分类任务数据集处理
    def __init__(self,path,smiles_field='Smiles',label_field=label,max_len=500,seed=1,batch_size=16,a=2,addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.df = self.df[[True if Chem.MolFromSmiles(smi) is not None else False for smi in self.df[smiles_field]]]
        self.seed = seed
        self.batch_size = batch_size
        self.a = a
        self.addH = addH

    def get_data(self):
        '''随机拆分数据集 random'''
        # data = self.df        
        # train_idx = []
        # idx = data.sample(frac=0.8).index
        # train_idx.extend(idx)
        # train_data = data[data.index.isin(train_idx)]

        # data = data[~data.index.isin(train_idx)]
        # test_idx = []
        # idx = data[~data.index.isin(train_data)].sample(frac=0.5).index
        # test_idx.extend(idx)
        # test_data = data[data.index.isin(test_idx)]

        # val_data = data[~data.index.isin(train_idx+test_idx)]

        '''按分子骨架拆分数据集,scaffold_split'''

        data = self.df
        #data = data.dropna()
        train_ids, val_ids, test_ids = scaffold_split(self.df, sizes=(0.8, 0.1, 0.1), balanced=True,seed=self.seed)
        data = data.fillna(666)
        train_data = self.df.iloc[train_ids]
        val_data = self.df.iloc[val_ids]
        test_data = self.df.iloc[test_ids]
        df_train_data = pd.DataFrame(train_data)
        df_test_data = pd.DataFrame(test_data)
        df_val_data = pd.DataFrame(val_data)
        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (df_train_data[self.smiles_field], df_train_data[self.label_field]))

        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(batch_size=self.batch_size, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((df_test_data[self.smiles_field], df_test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((df_val_data[self.smiles_field], df_val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).cache().prefetch(100)

        return self.dataset1 + self.dataset2 + self.dataset3

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        y = np.array(label).astype('int64')

        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])

        return x, adjoin_matrix , y


class Graph_Regression_Dataset(object):  #  图回归任务数据集处理
    def __init__(self,path,smiles_field='Smiles',label_field=label,seed=1,batch_size=32,a=1,max_len=500,normalize=True,addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.df = self.df[[True if Chem.MolFromSmiles(smi) is not None else False for smi in self.df[smiles_field]]]
        self.seed = seed
        self.batch_size = batch_size
        self.a = a
        self.addH = addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min)-0.5
            self.value_range = self.max-self.min

    def get_data(self):
        '''随机拆分数据集 random'''
        data = self.df        
        train_idx = []
        idx = data.sample(frac=0.8).index
        train_idx.extend(idx)
        train_data = data[data.index.isin(train_idx)]

        data = data[~data.index.isin(train_idx)]
        test_idx = []
        idx = data[~data.index.isin(train_data)].sample(frac=0.5).index
        test_idx.extend(idx)
        test_data = data[data.index.isin(test_idx)]

        val_data = data[~data.index.isin(train_idx+test_idx)]

        '''按分子骨架拆分数据集,scaffold_split'''
        # data = self.df
        # train_ids, val_ids, test_ids = scaffold_split(self.df, sizes=(0.8, 0.1, 0.1), balanced=True,seed=self.seed)
        # train_data = self.df.iloc[train_ids]
        # val_data = self.df.iloc[val_ids]
        # test_data = self.df.iloc[test_ids]

        df_train_data = pd.DataFrame(train_data)
        df_test_data = pd.DataFrame(test_data)
        df_val_data = pd.DataFrame(val_data)   

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (df_train_data[self.smiles_field], df_train_data[self.label_field]))

        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(batch_size=self.batch_size, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((df_test_data[self.smiles_field], df_test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((df_val_data[self.smiles_field], df_val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).cache().prefetch(100)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        y = np.array(label).astype('float32')

        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , y


class Inference_Dataset(object):
    def __init__(self,sml_list,max_len=500,addH=True):
        self.vocab = str2num
        self.devocab = num2str
        self.sml_list = [i for i in sml_list if len(i)<max_len]
        self.addH =  addH

    def get_data(self):

        self.dataset = tf.data.Dataset.from_tensor_slices((self.sml_list,))
        self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]),tf.TensorShape([None]))).cache().prefetch(20)

        return self.dataset

    def numerical_smiles(self, smiles):
        smiles_origin = smiles
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        return x, adjoin_matrix,[smiles], atoms_list

    def tf_numerical_smiles(self, smiles):
        x,adjoin_matrix,smiles,atom_list = tf.py_function(self.numerical_smiles, [smiles], [tf.int64, tf.float32,tf.string, tf.string])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        return x, adjoin_matrix,smiles,atom_list



# class multi_task_dataset(object):
#     def __init__(self,path_list,smiles_field,label_field,max_len=100,addH=True):
#         self.vocab = str2num
#         self.smiles_field = smiles_field
#         self.label_field = label_field
#         self.devocab = num2str
#         self.addH =  addH
#         self.pathlist = path_list
#
#     def get_data(self):
#         x_train_list = []
#         y_train_list = []
#         mask_train_list=[]
#         test_dataset_list = []
#         for i,path in enumerate(self.pathlist):
#             data = pd.read_csv(path,sep='\t')
#             lengths = [0, 25, 50, 75, 100]
#             train_idx = []
#             for ii in range(4):
#                 idx = data[(data[self.smiles_field].str.len() >= lengths[ii]) & (
#                         data[self.smiles_field].str.len() < lengths[ii + 1])].sample(frac=0.8).index
#                 train_idx.extend(idx)
#             data1 = data[data.index.isin(train_idx)].copy()
#             data2 = data[~data.index.isin(train_idx)].copy()
#             x_train_list += data1[self.smiles_field].tolist()
#             y_train = -np.ones((len(data1),len(self.pathlist))).astype('float32')
#             mask_train = np.zeros((len(data1),len(self.pathlist))).astype('float32')
#
#             y_train[:,i] = np.array(data1[self.label_field])
#             mask_train[:, i] = 1
#
#             y_train_list.append(y_train)
#             mask_train_list.append(mask_train)
#
#             x_test = data2[self.smiles_field].tolist()
#             y_test = -np.ones((len(data2),len(self.pathlist))).astype('float32')
#             y_test[:,i] = np.array(data2[self.label_field])
#             mask_test = np.zeros((len(data2),len(self.pathlist))).astype('float32')
#             mask_test[:, i] = 1
#             test_dataset_list.append(tf.data.Dataset.from_tensor_slices((x_test,y_test,mask_test)).map(self.tf_numerical_smiles).padded_batch(256,
#                                                                 padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]),
#                                                                 tf.TensorShape([None]),tf.TensorShape([None]))).cache().prefetch(100))
#
#         y_train_list = np.concatenate(y_train_list,axis=0)
#         mask_train_list = np.concatenate(mask_train_list,axis=0)
#
#         dataset1 = tf.data.Dataset.from_tensor_slices((x_train_list,y_train_list,mask_train_list))
#         dataset1 = dataset1.map(self.tf_numerical_smiles).shuffle(200).padded_batch(64, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([len(self.pathlist)]),
#             tf.TensorShape([len(self.pathlist)]))).cache().prefetch(100)
#         return dataset1, test_dataset_list
#
#     def numerical_smiles(self, smiles,y,y_mask):
#         smiles = smiles.numpy().decode()
#         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
#         atoms_list = ['<global>'] + atoms_list
#         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#         temp = np.ones((len(nums_list),len(nums_list)))
#         temp[1:,1:] = adjoin_matrix
#         adjoin_matrix = ((1-temp)*(-1e9)).astype('float32')
#         x = np.array(nums_list).astype('int64')
#         return x, adjoin_matrix,y,y_mask
#
#     def tf_numerical_smiles(self, smiles,y,y_mask):
#         x,adjoin_matrix,y, y_mask = tf.py_function(self.numerical_smiles, [smiles,y,y_mask], [tf.int64, tf.float32,tf.float32, tf.float32])
#         x.set_shape([None])
#         adjoin_matrix.set_shape([None,None])
#         y.set_shape([len(self.pathlist)])
#         y_mask.set_shape([len(self.pathlist)])
#         return x, adjoin_matrix,y, y_mask




class Graph_Regression_and_Pretraining_Dataset(object):
    def __init__(self,path,smiles_field='Smiles',label_field='Label',normalize=True,addH=True,max_len=100):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field] - self.min) / (self.max - self.min) - 0.5


    def get_data(self):
        data = self.df
        lengths = [0, 25, 50, 75, 100]
        train_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
            train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices((data1[self.smiles_field], data1[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([1]),tf.TensorShape([None,None]),tf.TensorShape([None]),tf.TensorShape([None]))).cache().shuffle(100).prefetch(100)
        self.dataset2 = tf.data.Dataset.from_tensor_slices((data2[self.smiles_field], data2[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([1]),tf.TensorShape([None,None]), tf.TensorShape([None]),tf.TensorShape([None]))).cache().prefetch(100)
        return self.dataset1,self.dataset2

    def numerical_smiles(self, smiles,label):

        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        x_true = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x_masked = np.array(nums_list).astype('int64')
        weight = weight.astype('int64')
        label = np.array([label]).astype('float32')
        return x_masked, label,  adjoin_matrix, x_true, weight


    def tf_numerical_smiles(self, smiles,label):
        x, label, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [smiles,label],
                                                     [tf.int64, tf.float32, tf.float32, tf.int64, tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        weight.set_shape([None])
        label.set_shape([None])

        return x, label,adjoin_matrix, y, weight