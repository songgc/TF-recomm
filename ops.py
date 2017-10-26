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
        cond = lambda i,_: tf.logical_and(tf.less(i, batch_size),tf.less(i,tf.size(user_batch)))
        # sum_y = tf.Variable([])
        sum_y = tf.TensorArray(tf.float32,size=batch_size)
        nonzero_sqrt = tf.TensorArray(tf.float32, size=batch_size)
        embd_y    = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_y")
        # transpose from [item_num, dim] to [dim, item_num]
        print("embd_y:{}".format(embd_y))
        embd_y = tf.transpose(embd_y)
        def sumy(i,sum):
            #mask shape is [dim, item_num]
            umask  =tf.nn.embedding_lookup(rmat_batch,user_batch) #get all user rows
            mask = tf.tile(tf.reshape(tf.gather(umask,i), (1, -1)), (dim, 1))
            mask= tf.transpose(tf.nn.embedding_lookup(tf.transpose(mask), item_batch))
            print("mask:{}".format(mask))
            mat = tf.reduce_sum(tf.matmul(embd_y,mask,transpose_a=False,transpose_b=True),axis=1)
            #sum(y)*|rating(u)|^(-1/2)
            mat = tf.multiply(mat,tf.pow(tf.add(tf.cast(tf.count_nonzero(tf.gather(mask,0)),tf.float32),tf.constant(0.00001)),-0.5))
            print("mat:{}".format(mat))
            # sum=tf.concat([sum,mat],axis=0)
            sum=sum.write(i,mat)
            i = tf.add(i,1)
            return i, sum

        #sum shape would be finally be [user_num,dim]

        idx,sum_y= tf.while_loop(cond, sumy, [i,sum_y])#,shape_invariants=[i.get_shape(), tf.TensorShape([None])])
        sum_y=sum_y.stack()
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.device(device):
        embd_user = tf.add(embd_user,sum_y)
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        print("infer:{}".format(infer))
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        #embd_item shape is [item_num,dim] sum_y = [user_num,dim],final shap would be [user_num,item_num]
        print("infer:{}".format(infer))
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
    return infer, regularizer

def optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        print("cost:{}".format(cost))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op
