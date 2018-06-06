import pandas as pd
import numpy as np
import tensorflow as tf
import random
import pickle


def main():
    print(tf.__version__)

    train = pd.read_csv('input/train.csv.zip')

    type(train)
    train.dtypes

    train.iloc[0, 1]

    train.iloc[0, 1]

    toxic = train.loc[train.toxic == 1]

    print(train.shape, toxic.shape)

    toxic.head()

    toxic.iloc[2, 1]

    toxic.iloc[3, 1]

    # construct a vocabulary as a dictionary: {word: integer}
    vocab = {'__none__': 0}
    for index, row in train.iterrows():
        comment = row['comment_text']
        if index > 0 and index % 50000 == 0:
            print('%d rows done, size of vocab = %d' % (index, len(vocab)))
        words = comment.split()
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)

    # map the word to an integer for all the words
    # build the X
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

    test = pd.read_csv('input/test.csv.zip')
    Xt = []
    for index, row in test.iterrows():
        comment = row['comment_text']
        if index > 0 and index % 50000 == 0:
            print('%d rows done' % (index))
        words = comment.split()
        x = []
        for word in words:
            x.append(vocab.get(word, 0))
        Xt.append(x)

    classes = train.columns.values[2:]
    Xt = np.array(Xt)

    y = train[classes].values

    print(len(X), len(y), y.shape)

    model = textCNN(vocab_size=len(vocab), embedding_size=4)

    model.fit(X, y)
    print("first fit done")

    model.fit(X, y)
    print("second fit done")

    model = textCNN(vocab_size=len(vocab), embedding_size=4)
    yp = model.predict(Xt)
    yp.shape
    s = pd.DataFrame(
        yp,
        columns=[
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
            'identity_hate'
        ])
    s['id'] = test['id']

    s.to_csv('submission.csv', index=False)

    s = pickle.load(open('weight.p', 'rb'))

    for name, values in s.items():
        print(name, values.shape)

    varss = tf.trainable_variables()

    type(varss)


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
        self.y = y
        self._train()

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
        logit = tf.nn.sigmoid(logit)
        preds = []
        print('here')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.load(sess)
            for c, Xb in enumerate(self._batch_gen(shuffle=False)):
                pred = sess.run(logit, feed_dict={self.inputs: Xb})
                if c % 10 == 0:
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

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            try:
                self.load(sess)
            except FileNotFoundError:
                print("no previous session! starting fresh")
            for c, (Xb, yb) in enumerate(self._batch_gen(shuffle=True)):
                loss, _ = sess.run(
                    [losst, opt_op], feed_dict={
                        self.inputs: Xb,
                        label: yb
                    })
                if c % 10 == 0:
                    print("batch %d train loss %.4f" % (c, loss))
                if c > 100:
                    break
            self.save(sess)

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
        pickle.dump(s, open('weight.p', 'wb'))

    def load(self, sess, path='weight.p'):
        s = pickle.load(open(path, 'rb'))
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

        netname = 'textCNN'
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])  # B,S

        with tf.variable_scope(netname):
            # embedding lookup
            print("Input:", self.inputs.get_shape().as_list())
            net = self.embedding_lookup(
                E, V, self.inputs, reuse=False)  # dim(net) = B,S,E
            print("after embedding lookup:", net.get_shape().as_list())

            net1 = self.conv1d(
                net, name='conv1', ngrams=3, stride=1, cin=E, nfilters=30)
            net2 = self.conv1d(
                net, name='conv2', ngrams=1, stride=1, cin=E, nfilters=10)
            net3 = self.conv1d(
                net, name='conv3', ngrams=5, stride=1, cin=E, nfilters=50)

            print("before maxpool net1: {} net2: {} net3: {}".format(
                net1.get_shape().as_list(),
                net2.get_shape().as_list(),
                net3.get_shape().as_list()))

            net1 = tf.reduce_max(net1, axis=1)
            net2 = tf.reduce_max(net2, axis=1)
            net3 = tf.reduce_max(net3, axis=1)

            print("after maxpool net1: {} net2: {} net3: {}".format(
                net1.get_shape().as_list(),
                net2.get_shape().as_list(),
                net3.get_shape().as_list()))

            net = tf.concat([net1, net2, net3], axis=1)
            print("after concat:", net.get_shape().as_list())
            #print("after conv1d:",net.get_shape().as_list())

            net = self.fc(net, 'fc', cin=90, cout=6, activation=None)
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
        filters = tf.get_variable(
            name='%s_filters' % name,
            shape=[ngrams, cin, nfilters],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )  # filter variable
        net = tf.nn.conv1d(net, filters, stride, padding='VALID')
        if activation == 'relu':
            net = tf.nn.relu(net)
        return net

    def embedding_lookup(self, E, V, x, reuse=False):
        # x is the key
        with tf.variable_scope('embedding', reuse=reuse):

            # initialize the embedding matrix variable
            w = tf.get_variable(
                name='w',
                shape=[V - 1, E],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer()
            )  # embedding matrix
            # w.name = 'textCNN/embedding/w:0'
            self.em = w
            c = tf.zeros([1, E])
            w = tf.concat([c, w], axis=0)

            # embedding lookup
            x = tf.cast(x, tf.int32)
            x = tf.nn.embedding_lookup(
                w, x, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _batch_gen(self, shuffle=True):
        X, y = self.X, self.y
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
        maxlen = max([len(i) for i in X])
        for x in X:
            xn = x + [0] * (maxlen - len(x))
            assert len(xn) == maxlen
            Xn.append(xn)
        return np.array(Xn)

    def _predict(self):
        pass

if __name__ == "__main__":
        main()
