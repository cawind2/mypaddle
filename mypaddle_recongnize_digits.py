# -*- coding:utf8 -*-
# digits
from __future__ import print_function

import os
from PIL import Image
import numpy
import paddle
import paddle.fluid as fluid

BATCH_SIZE = 64
PASS_NUM = 5


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def multilayer_perceptron(img, label):
    img = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    return loss_net(hidden, label)


def softmax_regression(img, label):
    return loss_net(img, label)


def convolutional_neural_network(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train(nn_type,
          use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # X是输入：MNIST图片是28×28 的二维图像
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # Label 是图片的真实标签：Label=(l0,l1,…,l9)也是10维，但只有一维为1，其他都为0。
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    #最简单的Softmax回归模型是先将输入层经过一个全连接层得到特征，然后直接通过 softmax 函数计算多个类别的概率并输出
    # 分别表示该样本属于这 N 个类别的概率
    if nn_type == 'softmax_regression':
        net_conf = softmax_regression
    #多层感知机(Multilayer Perceptron, MLP)
    #Softmax回归模型采用了最简单的两层神经网络，即只有输入层和输出层，因此其拟合能力有限。为了达到更好的识别效果，
    # 我们考虑在输入层和输出层中间加上若干个隐藏层
    elif nn_type == 'multilayer_perceptron':
        net_conf = multilayer_perceptron
    else:
    # LeNet-5是一个较简单的卷积神经网络。图4显示了其结构：输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，
    # 最后使用softmax分类作为输出层
        net_conf = convolutional_neural_network

    prediction, avg_loss, acc = net_conf(img, label)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        acc_set = []
        avg_loss_set = []
        for test_data in train_test_reader():
            acc_np, avg_loss_np = exe.run(
                program=train_test_program,
                feed=train_test_feed.feed(test_data),
                fetch_list=[acc, avg_loss])
            acc_set.append(float(acc_np))
            avg_loss_set.append(float(avg_loss_np))
        # get test acc and loss
        acc_val_mean = numpy.array(acc_set).mean()
        avg_loss_val_mean = numpy.array(avg_loss_set).mean()
        return avg_loss_val_mean, acc_val_mean

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    exe.run(fluid.default_startup_program())
    main_program = fluid.default_main_program()
    epochs = [epoch_id for epoch_id in range(PASS_NUM)]

    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(
                main_program,
                feed=feeder.feed(data),
                fetch_list=[avg_loss, acc])
            if step % 100 == 0:
                print("Pass %d, Batch %d, Cost %f" % (step, epoch_id,
                                                      metrics[0]))
            step += 1
        # test for epoch
        avg_loss_val, acc_val = train_test(
            train_test_program=test_program,
            train_test_reader=test_reader,
            train_test_feed=feeder)

        print("Test with Epoch %d, avg_cost: %s, acc: %s" %
              (epoch_id, avg_loss_val, acc_val))
        lists.append((epoch_id, avg_loss_val, acc_val))
        if save_dirname is not None:
            fluid.io.save_inference_model(
                save_dirname, ["img"], [prediction],
                exe,
                model_filename=model_filename,
                params_filename=params_filename)

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
    print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))


def infer(use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    tensor_img = load_image(cur_dir + '/image/infer_3.png')

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             save_dirname, exe, model_filename, params_filename)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets)
        lab = numpy.argsort(results)
        print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])


def main(use_cuda, nn_type):
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits_" + nn_type + ".inference.model"

    # call train() with is_local argument to run distributed train
    train(
        nn_type=nn_type,
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)
    infer(
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)


if __name__ == '__main__':
    use_cuda = False
    #predict = 'softmax_regression' # uncomment for Softmax
    #predict = 'multilayer_perceptron' # uncomment for MLP
    predict = 'convolutional_neural_network'  # uncomment for LeNet5
    main(use_cuda=use_cuda, nn_type=predict)

