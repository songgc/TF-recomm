from __future__ import print_function

import tensorflow as tf
from numpy import float32,int32,float64
def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return  int32(int(fields[0])-1), int32(int(fields[1])-1), float32(fields[2]), float32( float(fields[3]) % 10)

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle::MovieCategory .
    """
    fields = line.strip().split("::")
    s= fields[1]
    year = s[s.rfind("(")+1:s.rfind(")")]
    return int32(int(fields[0])-1), s, int32(year), fields[2]

def parseUser(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return  int32(int(fields[0])-1), fields[1], int32(fields[2]),int32(fields[3]), fields[4]


def readRating(filename_queue,reader):
    _, value = reader.read(filename_queue)
    y = tf.py_func(parseRating, [value], [tf.int32,tf.int32, tf.float32,tf.float32])
    return y

def ratinglabel(filename_queue,reader):
    y= readRating(filename_queue,reader)
    return y[0],y[1], y[2]


def readMovie(filename_queue):
    reader = tf.TextLineReader()
    _,value = reader.read(filename_queue)
    y = tf.py_func(parseMovie, [value], [tf.int32,tf.string, tf.int32,tf.string])
    return y


def readUser(filename_queue):
    reader = tf.TextLineReader()
    _,value = reader.read(filename_queue)
    y = tf.py_func(parseUser, [value], [tf.int32,tf.string, tf.int32,tf.int32, tf.string])
    return y


def shuffleInputPipeline(filename_queue, reader, batch_size, read_threads, num_epochs=None):

    userid, itemid, rating= ratinglabel(filename_queue,reader)
    userid.set_shape([])
    itemid.set_shape([])
    rating.set_shape([])
    min_after_dequeue = batch_size*10
    capacity = min_after_dequeue + 3 * batch_size
    user_batch,item_batch,rate_batch= tf.train.shuffle_batch(
        [userid, itemid, rating], batch_size=batch_size, capacity=capacity,num_threads=read_threads,
        min_after_dequeue=min_after_dequeue)
    return user_batch, item_batch,rate_batch


def readSparse(filename_queue, reader, batch_size, read_threads, num_epochs=None):
    userid, itemid, rating= ratinglabel(filename_queue,reader)
    userid.set_shape([])
    itemid.set_shape([])
    rating.set_shape([])
    min_after_dequeue = batch_size*10
    capacity = min_after_dequeue + 3 * batch_size
    user_batch,item_batch,rate_batch= tf.train.shuffle_batch(
        [userid, itemid, rating], batch_size=batch_size, capacity=capacity,num_threads=read_threads,
        min_after_dequeue=min_after_dequeue)
    return user_batch, item_batch,rate_batch

