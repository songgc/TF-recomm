from pipeline import shuffleInputPipeline
import time
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

import dataio
import ops

np.random.seed(13575)

BATCH_SIZE = 100
USER_NUM = 6040
ITEM_NUM = 3952
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"

#timesvd constant
BIN_NUMBER=30



def clip(x):
    return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def get_data():
    df = dataio.read_process("/tmp/movielens/ml-1m/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    maxtime = max(df['st'])
    ut_mean = df.groupby(['user'])['st'].mean()
    return df_train, df_test,maxtime,ut_mean

def svd_with_pipe(samples_per_batch):
    trainfilequeue = tf.train.string_input_producer(
        ["/tmp/movielens/ml-1m/ratings.dat"], num_epochs=None, shuffle=False)
    testfilequeue =  tf.train.string_input_producer(
        ["/tmp/movielens/ml-1m/ratings.dat"], num_epochs=None, shuffle=False)
    reader = tf.TextLineReader()
    user_batch,item_batch, rate_batch = shuffleInputPipeline(trainfilequeue,reader, BATCH_SIZE, 10)
    testuser_batch,testitem_batch, testrate_batch = shuffleInputPipeline(testfilequeue,reader, BATCH_SIZE, 10)

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)


    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    testusers, testitems,testrates = sess.run([testuser_batch,testitem_batch,testrate_batch])
    errors = deque(maxlen=samples_per_batch)
    print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
    try:
        for i in range(EPOCH_MAX * samples_per_batch):
            start = time.time()
            users, items, rates  = sess.run([user_batch,item_batch,rate_batch])
            _, pred_batch = sess.run([train_op, infer],feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])

                pred_batch = sess.run(infer,feed_dict={
                        user_batch: testusers,
                        item_batch: testitems,
                    })
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - testrates, 2))
                end = time.time()
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)),
                                                   end - start))
                start = end


    except tf.errors.OutOfRangeError:
        print ('Done Training')
    finally :
        coord.request_stop()
    coord.join(threads)
    sess.close()
def svdplusplus(train, test):
    samples_per_batch = BATCH_SIZE

    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                         train["rate"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.ShuffleIterator([test["user"],
                                         test["item"],
                                         test["rate"]],
                                        batch_size=BATCH_SIZE)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])
    rmat_batch = tf.placeholder(tf.float32, shape=[USER_NUM,ITEM_NUM],name="rmat")
    onecount_sqrt_batch = tf.placeholder(tf.float32,shape=[USER_NUM],name = "onecount_sqrt")

    # infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
    #                                        device=DEVICE)
    infer, regularizer = ops.inference_svdplusplus(user_batch,item_batch,rmat_batch,user_num=USER_NUM,item_num=ITEM_NUM,batch_size=BATCH_SIZE,dim=DIM)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            rmat = np.zeros([USER_NUM,ITEM_NUM],dtype=np.float32)
            rmat[users,items]=float(1.0)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates,
                                                                   rmat_batch: rmat,
                                                                   })
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            # print("i:{},errors:{}".format(i,np.sqrt(np.mean(errors))))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                users, items, rates =next(iter_test)
                rmat = np.zeros([USER_NUM, ITEM_NUM], dtype=np.float32)
                rmat[users, items] = float(1.0)
                # print("i:{},users:{},items:{}".format(i,users,items))
                pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                        item_batch: items,
                                                        rmat_batch: rmat,
                                                        })
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end

def timesvdplusplus(train, test,binsize,ut_mean):
    samples_per_batch = BATCH_SIZE

    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                         train["rate"],
                                         train["st"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.ShuffleIterator([test["user"],
                                         test["item"],
                                         test["rate"],
                                        test["st"]],
                                        batch_size=BATCH_SIZE)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])
    time_batch = tf.placeholder(tf.int32, shape = [None])
    rmat_batch = tf.placeholder(tf.float32, shape=[USER_NUM,ITEM_NUM],name="rmat")
    tu_batch   = tf.placeholder(tf.int32, shape=[None])
    # infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
    #                                        device=DEVICE)
    infer, regularizer = ops.inference_timesvdplusplus(user_batch,item_batch,time_batch,rmat_batch,tu_batch,binsize,maxtime,user_num=USER_NUM,item_num=ITEM_NUM,batch_size=BATCH_SIZE,dim=DIM)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates,times = next(iter_train)
            times= times
            rmat = np.zeros([USER_NUM,ITEM_NUM],dtype=np.float32)
            rmat[users,items]=float(1.0)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates,
                                                                   time_batch: times,
                                                                   rmat_batch: rmat,
                                                                   tu_batch  : ut_mean[users],
                                                                   })
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            # print("i:{},errors:{}".format(i,np.sqrt(np.mean(errors))))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                users, items, rates,times =next(iter_test)
                rmat = np.zeros([USER_NUM, ITEM_NUM], dtype=np.float32)
                rmat[users, items] = float(1.0)
                # print("i:{},users:{},items:{}".format(i,users,items))
                pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                        item_batch: items,
                                                        time_batch: times,
                                                        rmat_batch: rmat,
                                                        tu_batch  : ut_mean[users],
                                                        })
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end


if __name__ == '__main__':
    df_train, df_test,maxtime,ut_mean = get_data()
    # print(len(df_train))
    binsize = maxtime/BIN_NUMBER +1
    svdplusplus(df_train, df_test)
    #svd_with_pipe(100)
    # timesvdplusplus(df_train,df_test,binsize,ut_mean)
    print("Done!")
