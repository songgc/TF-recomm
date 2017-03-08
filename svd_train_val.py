import tensorflow as tf
from pipeline import shuffleInputPipeline
import dataio
import numpy as np
from collections import deque
from six import next
import time
import ops

np.random.seed(13575)

BATCH_SIZE = 1000
USER_NUM = 6040
ITEM_NUM = 3952
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"


def clip(x):
    return np.clip(x, 1.0, 5.0)


def get_data():
    df = dataio.read_process("/tmp/movielens/ml-1m/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

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

def svd(train, test):
    samples_per_batch = len(train) // BATCH_SIZE

    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                         train["rate"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test["user"],
                                         test["item"],
                                         test["rate"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    _, train_op = ops.optimiaztion(infer, regularizer, rate_batch, learning_rate=0.15, reg=0.05, device=DEVICE)

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)),
                                                       end - start))
                start = end

        output_graph_def = tf.python.framework.graph_util.extract_sub_graph(sess.graph.as_graph_def(),
                                                                         ["svd_inference", "svd_regularizer"])
        tf.train.SummaryWriter(logdir="/tmp/svd", graph_def=output_graph_def)


if __name__ == '__main__':
    df_train, df_test = get_data()
    # svd(df_train, df_test)
    svd_with_pipe(100)
    print("Done!")
