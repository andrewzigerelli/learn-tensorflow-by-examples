import pandas as pd
import numpy as np
import tensorflow as tf
import random
import pickle
import os



def main():
    # stop tensorflow polluting stdout
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    train = pd.read_csv('input/train.csv.zip')
    toxic = train.loc[train.toxic == 1]

    print("train size,  toxic size")
    print(train.shape, toxic.shape)

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

    # map the word to an integer for all the words
    # build the X
    try:
        X = load_obj("word2vec")
    except FileNotFoundError:
        X = []
        for index, row in train.iterrows():
            comment = row['comment_text']
            if index > 0 and index % 50000 == 0:
                print('%d rows done' % (index))
            words = comment.split()
            x = []
            for word in words:
                x.append(vocab.get(word, 0))
            X.append(x)
        save_obj(X, "word2vec")

    # test data
    test = pd.read_csv('input/test.csv.zip')
    Xt = []
    try:
        Xt = load_obj("test_data")
    except FileNotFoundError:
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

    classes = train.columns.values[2:]
    y = train[classes].values

    model = textCNN(vocab_size=len(vocab), embedding_size=4,batch_size=128)

    print("calling fit====================================")
    model.fit(X, y)
    print("done fit=======================================")


class textCNN:
    def __init__(self, **params):
        # params is a dictionary
        self.params = params
        self.y = None

    def fit(self, X, y):
        # X is a list of comments. [[fxxk, ..], [something, ..]]
        # len(X) == N == len(y)
    # y is a numpy array (N, 6)
        self.X = np.array(X)
        self.determine_pad()
        self.y = y
        self.label, self.losst, self.opt_op= self._train()

        # create tf Saver for movidius
        saver = tf.train.Saver()

        # stop polluting stdout
        tf.logging.set_verbosity(tf.logging.FATAL)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            try:
                self.load(sess)
                print("loading previous session")
            except FileNotFoundError:
                print("no previous session! starting fresh")
                for c,(Xb,yb) in enumerate(self._batch_gen(shuffle=True)):
                    loss,_ = sess.run([self.losst,self.opt_op],feed_dict={self.inputs:Xb, self.label:yb})
                    if c%10 == 0:
                        pass
                        # print("batch %d train loss %.4f"%(c,loss))
                    if c>100:
                        break
            self.save(sess)

            # save using tensorflow format for movidius
            graph_loc = "."
            save_path = saver.save(sess, graph_loc + "/graphs/toxic_textcnn_model")
            w = self.em
            x = tf.cast(Xb, tf.int32)
            save_obj(Xb, "Xb")
            lookup = tf.nn.embedding_lookup(
                w, x, name='word_vector')  # (N, T, M) or (N, M)
            lookup = sess.run(lookup)
            w = sess.run(w)
            print("last embedding")
            # print(lookup)
            # print(type(lookup))
            # print(w)
            print("w shape", w.shape)
            print("x shape", x.shape)
            print("lookup shape", lookup.shape)
            print("DONE with self save")


    def _fit(self, X, y):
        # X is a list of comments. [[fxxk, ..], [something, ..]]
        # len(X) == N == len(y)
        # y is a numpy array (N, 6)
        self.X = np.array(X)
        self.y = y
        return self._train()

    def predict(self, X):
        print('predict')
        # X is a list of comments. [[fxxk, ..], [something, ..]]
        self.X = X  # Xt means X for test data
        return self.predictx()

    def predictx(self):
        tf.reset_default_graph()
        self.params['epochs'] = 1
        # build a tf computing graph
        logit = self._build()
        # logit = tf.nn.sigmoid(logit, name="output")
        logit = tf.nn.sigmoid(logit)
        preds = []
        print('predicting:')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # self.load(sess)

            # for movidius, use tensorflow saver
            # restore trained network
            saver = tf.train.Saver()
            graph_loc = "."
            saver.restore(sess, graph_loc + "/graphs/toxic_textcnn_model")
            saver.save(sess, graph_loc + "/graphs/toxic_textcnn_inference")
            for c, Xb in enumerate(self._batch_gen(shuffle=False)):
                pred = sess.run(logit, feed_dict={self.inputs:Xb})
                if c % 100 == 0:
                    print("batch %d predicted" % (c))
                preds.append(pred)
        preds = np.vstack(preds)
        return preds

    def _train(self):
        tf.reset_default_graph()
        # build a tf computing graph
        logit = self._build()
        label = tf.placeholder(dtype=tf.int32, shape=[None, 6])  # B,classes
        losst = self.get_loss(logit, label)
        opt_op = self.get_opt(losst)
        return label, losst, opt_op

        # refactor to return tensor object
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     sess.run(tf.local_variables_initializer())
        #     try:
        #         self.load(sess)
        #         print("loading previous session")
        #     except FileNotFoundError:
        #         print("no previous session! starting fresh")
        #         for c,(Xb,yb) in enumerate(self._batch_gen(shuffle=True)):
        #             loss,_ = sess.run([losst,opt_op],feed_dict={self.inputs:Xb, label:yb})
        #             if c%10 == 0:
        #                 print("batch %d train loss %.4f"%(c,loss))
        #             if c>100:
        #                 break
        #     self.save(sess)

        #     # save using tensorflow format for movidius
        #     print("saving trained model")
        #     saver = tf.train.Saver()
        #     graph_loc = "."
        #     save_path = saver.save(sess, graph_loc + "/graphs/toxic_textcnn_model")

    def get_opt(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        return opt.minimize(loss)

    def get_loss(self, logit, label):
        # build the loss tensor
        label = tf.cast(label, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logit, labels=label)
        return tf.reduce_mean(loss)

    def save(self, sess):
        varss = tf.trainable_variables()
        s = {}  # var.name => var.value: a numpy array
        for var in varss:
            val = sess.run(var)
            s[var.name] = val
            print(type(val))
            print(type(var))
        save_obj(s, 'weight')
        # pickle.dump(s, open('weight.pkl', 'wb'))

    def load(self, sess, path='weight'):
        # s = pickle.load(open(path, 'rb'))
        s = load_obj(path)
        varss = tf.trainable_variables()
        for var in varss:
            value = s[var.name]
            assign_op = var.assign(value)
            sess.run(assign_op)

    def _build(self):

        # B is the batch size
        # S is the sequence length
        # E is the embedding size: 16 by default
        # V is the vocab size
        # embedding space will be: V x E
        # self.inputs => Xb
        E = self.params.get('embedding_size', 16)
        V = self.params['vocab_size']
        S = self.S

        netname = 'textCNN'
        self.inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, S], name="input")  # B,S

        with tf.variable_scope(netname):
            # embedding lookup
            print("Input:", self.inputs.get_shape().as_list())
            net = self.embedding_lookup(
                E, V, self.inputs, reuse=False)  # dim(net) = B,S,E
            print("after embedding lookup:", net.get_shape().as_list())

            net1 = self.conv1d(
                net, name='conv1', ngrams=3, stride=1, cin=E, nfilters=30)
            print("after conv1d", net1.get_shape().as_list())
            # net2 = self.conv1d(
            #     net, name='conv2', ngrams=1, stride=1, cin=E, nfilters=10)
            # net3 = self.conv1d(
            #     net, name='conv3', ngrams=5, stride=1, cin=E, nfilters=50)

            # net1 = tf.expand_dims(net1, -1)
            # print("after expand_dim", net1.get_shape().as_list())
            net1 = tf.nn.max_pool(net1, [1, 1411, 4, 1], [1, 1, 1, 1], padding="VALID")
            print("after nn_max_pool", net1.get_shape().as_list())
            net1 = tf.squeeze(net1, [1, 2])
            print("after squeeze", net1.get_shape().as_list())
            # need to try to avoid max layer
            # net1 = tf.reduce_max(net1, axis=1)
            # net1 = tf.reduce_max(net1, axis=1)
            # print("after maxpool net1", net1.get_shape().as_list())
            # print("before maxpool net1: {} net2: {} net3: {}".format(
            #     net1.get_shape().as_list(),
            #     net2.get_shape().as_list(),
            #     net3.get_shape().as_list()))

            # net1 = tf.reduce_max(net1, axis=1)
            # net2 = tf.reduce_max(net2, axis=1)
            # net3 = tf.reduce_max(net3, axis=1)

            # print("after maxpool net1: {} net2: {} net3: {}".format(
            #     net1.get_shape().as_list(),
            #     net2.get_shape().as_list(),
            #     net3.get_shape().as_list()))

            # net = tf.concat([net2, net3], axis=1)
            # print("after concat:", net.get_shape().as_list())
            # #print("after conv1d:",net.get_shape().as_list())

            net = self.fc(net1, 'fc', cin=30, cout=6, activation=None)
            print("after fc:", net.get_shape().as_list())
            return net

    def fc(self, net, name, cin, cout, activation='relu'):
        s = tf.get_variable(
            name=name,
            shape=[cin, cout],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        net = tf.matmul(net, s)
        if activation == 'relu':
            net = tf.nn.relu(net)
        return net

    def conv1d(self,
               net,
               name,
               ngrams,
               stride,
               cin,
               nfilters,
               activation='relu'):
        # filters: kernel_size, cin, cout
        # filters = tf.get_variable(
        #     name='%s_filters' % name,
        #     shape=[ngrams, cin, nfilters],
        #     dtype=tf.float32,
        #     initializer=tf.contrib.layers.xavier_initializer()
        # )  # filter variable
        filters = tf.get_variable(
            name='%s_filters' % name,
            shape=[ngrams, cin, 1, nfilters],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )  # filter variable
        print("filter dim", filters.get_shape())
        print("net size", net.get_shape())
        strides = [1, 1, 1, 1] # for 2d
        # net = tf.nn.conv1d(net, filters, stride, padding='SAME')
        net = tf.nn.conv2d(net, filters, strides, padding='SAME')
        if activation == 'relu':
            net = tf.nn.relu(net)
        return net

    def embedding_lookup(self, E, V, x, reuse=False):
        # x is the key
        # rewrote to avoid tf.gather

        # rows = np.arange(x.get_shape().as_list()[0])
        # stack_list = [w[x[row], :] for row in rows]
        # embedding = tf.stack(stack_list)

        # return embedding



        print("key size lookup:", x.get_shape().as_list())
        with tf.variable_scope('embedding', reuse=reuse):

            # initialize the embedding matrix variable
            w = tf.get_variable(
                name='w',
                shape=[V - 1, E, 1],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer()
            )  # embedding matrix
            # w.name = 'textCNN/embedding/w:0'
            self.em = w
            c = tf.zeros([1, E])
            #w = tf.concat([c, w], axis=0)

            # embedding lookup
            x = tf.cast(x, tf.int32)
            x = tf.nn.embedding_lookup(
                w, x, name='word_vector')  # (N, T, M) or (N, M)
            # avoid tf.gather
            # rows = np.arange(x.get_shape().as_list()[1])
            # print("x shape", x.get_shape())
            # print("w shape", w.get_shape())
            # print("rows", x.get_shape().as_list()[1])
            # stack_list = [w[x[0, row], :] for row in rows]
            # embedding = tf.stack(stack_list, axis = 0)
            # embedding = tf.expand_dims(embedding, 0)
            print("embedding size", x.get_shape().as_list())
            return x

    def _batch_gen(self, shuffle=True):
        X, y = self.X, self.y
        print("in batch gen")
        print("X, ", X.shape)
        B = self.params.get('batch_size', 128)
        epochs = self.params.get('epochs', 10)
        ids = [i for i in range(len(X))]
        batches = len(X) // B + 1
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
                yield Xb

    def padding(self, X):
        Xn = []
        # maxlen = max([len(i) for i in X])
        maxlen = self.S
        for x in X:
            xn = x + [0] * (maxlen - len(x))
            assert len(xn) == maxlen
            Xn.append(xn)
        return np.array(Xn)

    def _predict(self):
        pass

    def determine_pad(self):
        # clips input sequences to length S
        lengths = [len(i) for i in self.X]
        self.S = max(lengths)
        print("m comment length", self.S)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
