from multiprocessing.spawn import prepare
import time

from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from sklearn import metrics
from models import GAT, HeteGAT, HeteGAT_multi
from utils import process

# 禁用gpu
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "False"
np.random.seed(5)
tf.random.set_seed(5)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

dataset = 'my'
featype = 'fea'
checkpt_file = 'check/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
print_interval=20
batch_size = 1
nb_epochs = 1500
patience = 300
drop_out=0.4
pos_weight=8.0
lr = 0.001  # learning rate
l2_coef = 0.001 # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [1000,1000]
n_heads = [2,1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.relu
model = HeteGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

# jhy data
import scipy.io as sio
import scipy.sparse as sp
import pandas as pd
from sklearn.model_selection import train_test_split
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def normalize_adj(adjacency):
  adjacency += sp.eye(adjacency.shape[0])*0.0001
  degree = np.array(adjacency.sum(1))
  d_hat = sp.diags(np.power(degree, -0.5).flatten())
  return (d_hat.dot(adjacency).dot(d_hat)+sp.eye(adjacency.shape[0])).tocoo()
def load_my_data():
    
    mm= sp.load_npz( './my/sameManager.npz')
    ii= sp.load_npz('./my/ii.npz')
    rpt=sp.load_npz('./my/rpt.npz')
    py= sp.load_npz('./my/py.npz')
    mm=normalize_adj(mm)
    rpt=normalize_adj(rpt+py)
    data=pd.read_csv('./my/RT_SC_B_features_v2.csv')
    X=data.drop(['year','Stkcd','IndustryCode','label','EquityNatureID'],axis=1)
    for column in list(X.columns[X.isna().sum() > 0]):
        mean_val = X[column].mean()
        X[column].fillna(mean_val, inplace=True)
    X=X.values
    from sklearn.preprocessing import minmax_scale
    X=minmax_scale(X)
    #X=process.preprocess_features(X)
    y=data['label'].values
    y=np.reshape(y,(len(y),1))
    train_ids,test_ids=train_test_split(np.array(range(len(y))),stratify=data['label'].values,random_state=5,test_size=0.2)
    train_mask_row=sample_mask(train_ids,len(y))
    train_ids,val_ids=train_test_split(train_ids,test_size=0.25,stratify=data['label'].values[train_ids],random_state=5)
    train_mask = sample_mask(train_ids, y.shape[0])
    val_mask = sample_mask(val_ids, y.shape[0])
    test_mask = sample_mask(test_ids, y.shape[0])
    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_ids.shape,
                                                                                          val_ids.shape,
                                                                                          test_ids.shape))
    truefeatures_list = [X,X]
    rownetworks=[mm.A,rpt.A]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data_dblp(path='./ACM3025.mat'):
    data = sio.loadmat(path)
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]

    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask


# use adj_list as fea_list, have a try~
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_my_data()
if featype == 'adj':
    fea_list = adj_list



import scipy.sparse as sp




nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]

# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')
with tf.Graph().as_default():
    with tf.compat.v1.name_scope('input'):
        ftr_in_list = [tf.compat.v1.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.compat.v1.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss=model.masked_sigmoid_cross_entropy(log_resh, lab_resh, msk_resh,weight=pos_weight)
    #loss,da,db,dc = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh,weight=pos_weight)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    prediction_logits,golden_label=model.masked_output(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.compat.v1.train.Saver()

    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0
        val_auc_val=0
        for epoch in range(nb_epochs):
            tr_step = 0
           
            tr_size = fea_list[0].shape[0]
            # ================   training    ============
            while tr_step * batch_size < tr_size:

                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: drop_out,
                       ffd_drop: drop_out}
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                _, loss_value_tr, pred_logits_tr,label_tr,acc_tr, att_val_train = sess.run([train_op, loss,prediction_logits, golden_label,accuracy, att_val],
                                                                   feed_dict=fd)
                train_loss_avg += loss_value_tr
                train_auc=metrics.roc_auc_score(label_tr[:,0],pred_logits_tr[:,0])
                train_acc_avg += train_auc
                tr_step += 1

            vl_step = 0
            vl_size = fea_list[0].shape[0]
            # =============   val       =================
            while vl_step * batch_size < vl_size:
                # fd1 = {ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size]}
                fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                       msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                       is_train: False,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}
          
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                loss_value_vl, logits_vl,golden_vl = sess.run([loss,prediction_logits,golden_label],
                                                 feed_dict=fd)
                val_loss_avg += loss_value_vl
                val_acc_avg += metrics.roc_auc_score(golden_vl[:,0],logits_vl[:,0])
                vl_step += 1
            # import pdb; pdb.set_trace()
            if epoch%print_interval==0:
                print('time:{},Epoch: {}, att_val: {}'.format(time.asctime(time.localtime(time.time())),epoch, np.mean(att_val_train, axis=0)))
                print('Training: loss = %.5f, auc = %.5f | Val: loss = %.5f, auc = %.5f' %
                    (train_loss_avg / tr_step, train_acc_avg / tr_step,
                    val_loss_avg / vl_step, val_acc_avg / vl_step))

            if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg / vl_step
                    vlss_early_model = val_loss_avg / vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn,
                          ', Max auc: ', vacc_mx)
                    print('Early stop model validation loss: ',
                          vlss_early_model, ', auc: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
            
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}
        
            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, logits_ts, label_test,jhy_final_embedding = sess.run([loss, prediction_logits, golden_label, final_embedding],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += metrics.roc_auc_score(label_test[:,0],logits_ts[:,0])
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test auc:', ts_acc / ts_step)
        sess.close()
