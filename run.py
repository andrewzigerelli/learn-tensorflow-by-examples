#! /usr/bin/env python3

import mvnc.mvncapi as mvnc
import sys
import numpy as np
import pickle


def main():
    path_to_networks = './graphs/'
    graph_filename = 'toxic_textcnn_inference.graph'

    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 0)
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    device = mvnc.Device(devices[0])
    try:
        device.open()
    except:
       print("couldn't open device")
       exit(0)

    print("Device opened sucessfully")

    #Load graph
    with open(path_to_networks + graph_filename, mode='rb') as f:
        graphFileBuff = f.read()

    # get input
    # construct a vocabulary as a dictionary: {word: integer}
    try:
        vocab = load_obj("vocab")
    except FileNotFoundError:
        vocab = {'__none__': 0}
        for index, row in train.iterrows():
            comment = row['comment_text']
            if index > 0 and index % 50000 == 0:
                print('%d rows done, size of vocab = %d' % (index, len(vocab)))
            words = comment.split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
        save_obj(vocab, "vocab")
    # test data
    try:
        Xt = load_obj("test_data")
    except FileNotFoundError:
        test = pd.read_csv('input/test.csv.zip')
        for index, row in test.iterrows():
            comment = row['comment_text']
            if index > 0 and index % 50000 == 0:
                print('%d rows done' % (index))
            words = comment.split()
            x = []
            for word in words:
                x.append(vocab.get(word, 0))
            Xt.append(x)
        save_obj(Xt, "test_data")
    Xt = np.array(Xt)

    model = textCNN(vocab_size=len(vocab), embedding_size=4, batch_size=1)
    input = model.predict(Xt)
    input = input.astype(np.float32) 
    print("Did embedding on cpu")

    graph = mvnc.Graph('graph')
    fifoIn, fifoOut = graph.allocate_with_fifos(device, graphFileBuff)

    print('Start download to NCS...')
    graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, input, 'user object')
    output, userobj = fifoOut.read_elem()
    print("Got here")
    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()
    device.close()
    exit(0)

    top_inds = output.argsort()[::-1][:5]

    print(''.join(['*' for i in range(79)]))
    print('inception-v3 on NCS')
    print(''.join(['*' for i in range(79)]))
    for i in range(5):
        print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

    print(''.join(['*' for i in range(79)]))
    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()
    device.close()
    print('Finished')


class textCNN:
    def __init__(self, **params):
        # params is a dictionary
        self.params = params
        self.y = None
        self.S = load_obj("S")

    def _batch_gen(self, shuffle=True):
        X, y = self.X, self.y
        B = self.params.get('batch_size', 128)
        epochs = self.params.get('epochs', 10)
        ids = [i for i in range(len(X))]
        batches = len(X) // B + bool(len(X) % B)
        print(epochs, batches)
        for epoch in range(epochs):
            if shuffle:
                random.shuffle(ids)
            for i in range(batches):
                idx = ids[i * B:(i + 1) * B]
                Xb = self.padding(X[idx])
                if y is not None:
                    yield Xb, y[idx]
                else:
                    yield Xb
            if (i + 1) * B < len(X):
                idx = ids[(i + 1) * B:len(X)]
                Xb = self.padding(X[idx])
                return Xb

    def padding(self, X):
        Xn = []
        # maxlen = max([len(i) for i in X])
        maxlen = self.S
        for x in X:
            xn = x + [0] * (maxlen - len(x))
            assert len(xn) == maxlen
            Xn.append(xn)
        return np.array(Xn)

    def predict(self, X):
        # X is a list of comments. [[fxxk, ..], [something, ..]]
        self.X = X  # Xt means X for test data
        return self.predictx()

    def predictx(self):
        self.params['epochs'] = 1
        for Xb in self._batch_gen(shuffle=False):
            Xb = self.embedding_lookup_np(Xb)
            return Xb

    def embedding_lookup_np(self, Xb):
        weight = load_obj("weight")
        w = weight["textCNN/embedding/w:0"]
        embedding = [np.take(w,Xb[i,:], axis=0) for i in range(Xb.shape[0])]
        embedding = np.array(embedding)
        return embedding


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
