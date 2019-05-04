
import tensorflow as tf
import os
import re
import csv
import io
import requests
import numpy as np
from tensorflow.python.framework import ops
from zipfile import ZipFile
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

def getData():
   temp = []
   with open("SpamClassified.csv", 'r') as data:

      csvReader = csv.reader(data)
      for row in data:
         temp.append(row)


   temp = [x.split('\t') for x in temp if len(x) >= 1]
   return temp

def cleanText(text):
   text = re.sub(r'([^\s\w]|_|[0-9])+', '', text)
   text = " ".join(text.split())
   text = text.lower()
   return text

def main():
   sess = tf.Session()

   batchSize = 250
   maxLength = 25
   sizeRNN = 10
   embeddingSize = 50
   minFrequency = 10
   R = 0.001
   probDropout = tf.placeholder(tf.float32)

   lossTrain = []
   lossTest = []
   accuracyTrain = []
   accuracyTest= []

   allData = getData()

   dataLabel = []
   dataTrain = []

   for x in allData:
      if len(x) > 1:
         dataLabel.append(x[0])
         dataTrain.append(cleanText(x[1]))

   vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor(maxLength, min_frequency=minFrequency)

   dataProcessed = np.array(list(vocabProcessor.fit_transform(dataTrain)))
   dataProcessed = np.array(dataProcessed)

   dataLabel = np.array([1 if x == 'ham' else 0 for x in dataLabel])

   shuffle = np.random.permutation(np.arange(len(dataLabel)))
   shuffleX = dataProcessed[shuffle]
   shuffleY = dataLabel[shuffle]

   xCutoff = int(len(shuffleY) * 0.75)
   trainX, testX = shuffleX[:xCutoff], shuffleX[xCutoff:]
   trainY, testY = shuffleY[:xCutoff], shuffleY[xCutoff:]
   vocabSize = len(vocabProcessor.vocabulary_)

   dataX = tf.placeholder(tf.int32, [None, maxLength])
   outputY = tf.placeholder(tf.int32, [None])
   embedMat = tf.get_variable("embedding_mat", shape=[vocabSize, embeddingSize], dtype=tf.float32,
                              initializer=None, regularizer=None, trainable=True, collections=None)

   embedOutput = tf.nn.embedding_lookup(embedMat, dataX)
   cell = tf.nn.rnn_cell.BasicRNNCell(num_units = sizeRNN)
   output, state = tf.nn.dynamic_rnn(cell, embedOutput, dtype=tf.float32)
   output = tf.nn.dropout(output, probDropout)
   output = tf.transpose(output, [1, 0, 2])

   last = tf.gather(output, int(output.get_shape()[0]) - 1)
   weight = tf.get_variable("weight", shape=[sizeRNN, 2], dtype=tf.float32, initializer=None, regularizer=None,
                            trainable=True, collections=None)

   bias = tf.get_variable("bias", shape=[2], dtype=tf.float32, initializer=None, regularizer=None, trainable=True,
                          collections=None)

   logitsOut = tf.nn.softmax(tf.matmul(last, weight) + bias)
   losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logitsOut, labels=outputY)
   loss = tf.reduce_mean(losses)
   accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logitsOut, 1), tf.cast(outputY, tf.int64)), tf.float32))

   optimizer = tf.train.RMSPropOptimizer(R)
   stepTrain = optimizer.minimize(loss)
   init_op = tf.global_variables_initializer()

   sess.run(init_op)
   for epoch in range(300):

      shuffle = np.random.permutation(np.arange(len(trainX)))

      trainX = trainX[shuffle]
      trainY = trainY[shuffle]

      numBatchs = int(len(trainX) / batchSize) + 1

      for i in range(numBatchs):

         xMin = i * batchSize
         xMax = np.min([len(trainX), ((i + 1) * batchSize)])

         batchX = trainX[xMin:xMax]
         batchY = trainY[xMin:xMax]

         trainDict = {dataX: batchX, outputY: batchY, probDropout:0.5}
         sess.run(stepTrain, feed_dict=trainDict)
         tempLoss, temp_train_acc = sess.run([loss, accuracy], feed_dict=trainDict)
         lossTrain.append(tempLoss)
         accuracyTrain.append(temp_train_acc)

      testDict = {dataX: testX, outputY: testY, probDropout:1.0}
      testLossTemp, testAccTemp = sess.run([loss, accuracy], feed_dict=testDict)
      lossTest.append(testLossTemp)
      accuracyTest.append(testAccTemp)

   print('Accuracy: ',format(np.mean(accuracyTest[-1])*100.0))


if __name__=="__main__":
    main()