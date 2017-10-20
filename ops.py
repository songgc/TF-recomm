import tensorflow as tf


def inference_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))

        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.device(device):
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
    return infer, regularizer


def inference_svdplusplus(user_batch,item_batch,rmat_batch,user_num,item_num,batch_size,dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))

        #svd++ factor set
        #y shape is [dim, item_num]
        y = tf.get_variable("embd_y", shape=[item_num,dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))

        i = tf.constant(0)
        cond = lambda i,_: tf.less(i, batch_size)
        sum_y = tf.TensorArray(dtype=tf.float32, size=batch_size)
        embd_y    = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_y")
        # transpose from [item_num, dim] to [dim, item_num]
        print("embd_y:{}".format(embd_y))
        embd_y = tf.transpose(embd_y)
        def sumy(i,sum):
            i = tf.add(i,1)
            #mask shape is [dim, item_num]
            mask = tf.tile(tf.reshape(tf.gather(rmat_batch, tf.gather(user_batch,i)), (1, -1)), (dim, 1))
            print("mask:{}".format(mask))
            mat = tf.reduce_sum(tf.multiply(embd_y,mask),axis=1)
            print("mat:{}".format(mat))
            sum.write(i,mat)
            return i, sum

        #sum shape would be finally be [user_num,dim]

        idx,sum_y= tf.while_loop(cond, sumy, [i,sum_y])
        print(sum_y)
        sum_y=sum_y.pack()
        print(sum_y)
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.device(device):
        #infer shape is [item_num
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        #infer need to add the  sum(y)*|rating(u)|^(-1/2)
        #embd_item shape is [item_num,dim] sum_y = [user_num,dim],final shap would be [user_num,item_num]
        r_embedd=tf.transpose(tf.matmul(embd_item,sum_y,transpose_a=False,transpose_b=True))
        infer = tf.add(infer, tf.reduce_sum(r_embedd,1))

        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
    return infer, regularizer

def optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op
