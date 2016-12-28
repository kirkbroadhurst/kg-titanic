import numpy as np
import csv
import sys


def cost(estimated, y, thetas=[], l=0):
    """
    Compute the cost of a neural network vs the expect / label output
    :param estimated: The output of the final layer of a neural network - the estimation
    :param y: The expected value for output of the neural network - the label
    :param thetas: Coefficients used in neural network (regularization)
    :param l: lambda value - regularization parameter
    :return: Cost of the network
    """

    # the number of observations
    m = y.shape[0]

    # take log of the estimated value so that the 'cost' of predicting 1 is 0, and the 'cost' of predicting zero -> inf
    # multiply by y, i.e. when y = 1 we should have estimated 1; an estimate closer to zero is high cost i.e. log(0)
    # -y * log(estimated)

    # the reverse applies for y = 0; we want 'high cost' when y = 0 and estimation -> 1; so use log(1 - est) -> inf.
    # and multiply by 1 - y, i.e. 1 when y == 0

    gap = np.multiply(-y, np.log(estimated)) - np.multiply(1-y, np.log(1-estimated))
    j = 1.0 / m * np.sum(gap)

    # if thetas are supplied for regularization, return sum of squares
    for t in thetas:
        # remove the bias / constant term
        t_ = t[:, 1:]
        j += l/(2.0*m) * np.sum(np.multiply(t_, t_))

    return j


def load_csv(file_name):
    """
    Load a csv file into a numpy matrix of floats
    :param file_name: file name
    :return:
    """
    with open(file_name, 'r') as f:
        data_iter = csv.reader(f, delimiter=',')
        header = data_iter.next()
        data = [data for data in data_iter]
    return data


def predict(X, syn0, syn1):
    x_ = np.matrix(np.empty((X.shape[0], X.shape[1]+1)))
    x_[:, 0] = 1
    x_[:, 1:] = X
    l1 = 1 / (1 + np.exp(-(np.dot(x_, syn0))))

    # gotta add that bias term
    l1_ = np.matrix(np.empty((l1.shape[0], l1.shape[1] + 1)))
    l1_[:, 0] = 1
    l1_[:, 1:] = l1
    l2 = 1 / (1 + np.exp(-(np.dot(l1_, syn1))))
    return (l2 >= 0.5).astype(float)


def score(X, y, syn0, syn1):
    predicted = predict(X, syn0, syn1)
    return np.sum(1.0 * y[predicted == y]) / np.sum(1.0 * y)


def iterate(X, y, s, reg, iterations):
    m = X.shape[0]
    reg_rate = 1.0 if reg is None else reg
    rate = 0.03
    syn0 = s[0] if s[0] is not None else 2*np.random.random((X.shape[1]+1, 10)) - 1
    syn1 = s[1] if s[1] is not None else 2*np.random.random((11, 1)) - 1
    x_ = np.matrix(np.empty((X.shape[0], X.shape[1]+1)))
    x_[:, 0] = 1
    x_[:, 1:] = X
    for j in xrange(iterations):
        '''
        # for batches let's chunk up the x_ & y arrays, and change m
        batch = 10
        indices = range(j, j + batch)
        xx = np.matrix(np.empty((batch, x_.shape[1])))
        xx[:, :] = [x_[ix % m] for ix in indices]

        yy = np.matrix(np.empty((batch, 1)))
        yy[:, :] = [y[ix % m] for ix in indices]

        syn0, syn1, l2 = descend(xx, yy, syn0, syn1, reg_rate, rate, batch)
        '''

        syn0, syn1, l2 = descend(x_, y, syn0, syn1, reg_rate, rate, m)
        if j % 100 == 0:
            #syn0, syn1, l2 = descend(x_, y, syn0, syn1, reg_rate, rate, m)
            print 'reg_rate {0}'.format(reg_rate), 'cost =', cost(l2, y, [syn0, syn1], reg_rate),
            sys.stdout.flush()
            print '\r',

    #syn0, syn1, l2 = descend(x_, y, syn0, syn1, reg_rate, rate, m)
    print 'reg_rate {0}'.format(reg_rate), 'cost =', cost(l2, y, [syn0, syn1], reg_rate), \
        'score = {0:.0f}%'.format(100 * score(X, y, syn0, syn1))

    return syn0, syn1


def descend(x_, y, syn0, syn1, reg_rate, rate, m):
    l1 = 1 / (1 + np.exp(-(np.dot(x_, syn0))))

    # gotta add that bias term
    l1_ = np.matrix(np.empty((l1.shape[0], l1.shape[1] + 1)))
    l1_[:, 0] = 1
    l1_[:, 1:] = l1

    l2 = 1 / (1 + np.exp(-(np.dot(l1_, syn1))))
    l2_delta = np.multiply((y - l2), (np.multiply(l2, (1 - l2))))
    l1_delta = np.multiply(l2_delta.dot(syn1.T), np.multiply(l1_, (1 - l1_)))
    syn1 += rate / m * (l1_.T.dot(l2_delta) - float(reg_rate) * syn1)

    # got to drop off the error for the bias term (don't care about this
    syn0 += rate / m * (x_.T.dot(l1_delta[:, 1:]) - float(reg_rate) * syn0)
    return syn0, syn1, l2


def run_test():
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    result = iterate(X, y)
    pass


def column_values(data, offset=0):
    """

    :param data:
    :param offset:  use -1 offset to adjust the index/key for each column on the test dataset
    :return:
    """
    values = dict([(i + offset, np.unique([d[i] for d in data])) for i in [2, 4, 11]])
    values[10 + offset] = np.unique([d[10][0] for d in data if len(d[10]) > 0])
    return values


def train(loops=10, iterations=10000):
    data = load_csv('data/train.csv')

    y = np.matrix([d[1] for d in data]).T.astype('float')

    # categorical/non-continuous variables: 2 (Pclass), 4 (sex), 10 (Cabin), 11 (Embarked)
    # split the 'cabin' variable on the letter component
    # maybe: 6 (SibSp), 7 (Parch)
    d_values = column_values(data)

    cols = range(2, len(data[0]))

    # make a list of lists for each row before flattening. Doing the nested lookup (for v in d_values) is hard.
    # remove the name and ticket
    X = [[[d[c] == v for v in d_values[c]] if d_values.has_key(c) else [d[c]] for c in cols if c not in [3, 8]] for d in data]
    X = np.matrix([[float(i) if i != '' else 0.0 for items in row for i in items] for row in X]).astype(float)

    for i in range(1, loops):
        #for reg in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]: #, 10.0]:
        #for reg in [0, 0.01, 0.03]:
        for reg in [0.01]:
            s0 = None
            s1 = None
            try:
                [s0, s1] = np.load('theta_{0}.npy'.format(reg))
            except:
                print 'couldn\'t load'
                pass

            result = iterate(X, y, [s0, s1], reg, iterations)
            np.save('theta_{0}'.format(reg), result)


def test():
    data = load_csv('data/test.csv')
    # in the test dataset we have one less column (1/survived)
    # everything is bumped down one column

    # categorical/non-continuous variables: 2 (Pclass), 4 (sex), 10 (Cabin), 11 (Embarked)
    # split the 'cabin' variable on the letter component
    # maybe: 6 (SibSp), 7 (Parch)
    train_data = load_csv('data/train.csv')
    d_values = column_values(train_data, -1)

    cols = range(1, len(data[0]))

    # make a list of lists for each row before flattening. Doing the nested lookup (for v in d_values) is hard.
    # remove the name and ticket

    # to ensure data is the same shape we will load the train data then remove it.
    X = [[[d[c] == v for v in d_values[c]] if c in d_values else [d[c]] for c in cols if c not in [2, 7]] for d in data]
    X = [[float(i) if i != '' else 0.0 for items in row for i in items] for row in X]
    X = np.matrix(X).astype(float)
    m = X.shape[0]

    for reg in [0, 0.01, 0.03]:
        [s0, s1] = np.load('theta_{0}.npy'.format(reg))
        result = np.matrix(np.empty((m, 2)))
        result[:, 0] = np.matrix(np.array([d[0] for d in data])).T
        result[:, 1] = predict(X, s0, s1)
        result = result.astype(int)
        with open('predict_{0}'.format(reg), 'wb') as f:
            f.write('PassengerId,Survived\n')
            np.savetxt(f, result, fmt='%d', delimiter=',')


if __name__ == "__main__":
    '''
    train_data =load_csv('data/train.csv')
    d = train_data[0]
    d_values = column_values(train_data)
    [[d[c] == v for v in d_values[c]] if d_values.has_key(c) else [d[c]] for c in cols if c not in [3, 8]]

    t1 = load_csv('data/test.csv')[0]
    '''
    test()
    #train(50)
