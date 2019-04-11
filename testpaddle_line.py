# -*- coding:utf8 -*-
# Linear Regression
from __future__ import print_function

import sys

import math
import numpy as np
import six

import paddle
import paddle.fluid as fluid


MY_TRAIN_DATA = None
MY_TEST_DATA = None
FILENAME = './test.csv'


def load_data(filename, feature_num=2, ratio=0.8):
    global MY_TRAIN_DATA, MY_TEST_DATA
    if MY_TRAIN_DATA is not None and MY_TEST_DATA is not None:
        return

    data = np.fromfile(filename, sep=' ')
    data = data.reshape(data.shape[0] // feature_num, feature_num)
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0) / data.shape[0]
    # feature_range(maximums[:-1], minimums[:-1])
    for i in six.moves.range(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    offset = int(data.shape[0] * ratio)
    MY_TRAIN_DATA = data[:offset]
    MY_TEST_DATA = data[offset:]


def train():

    global MY_TRAIN_DATA, FILENAME
    load_data(FILENAME)

    def reader():
        for d in MY_TRAIN_DATA:
            yield d[:-1], d[-1:]

    return reader


def test():

    global MY_TEST_DATA, FILENAME
    load_data(FILENAME)

    def reader():
        for d in MY_TEST_DATA:
            yield d[:-1], d[-1:]

    return reader

# For training test cost
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]


def save_result(points1, points2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')


def main():
    # 生成的模拟数据是y=a*x+b
    # x是第一列，输入数据
    # y是第二列，输出数据，预测数据
    feature_names = ['x', 'y']

    feature_num = len(feature_names)

    #filename = './test.csv'
    #data = np.fromfile(filename, sep=' ')
    #data = data.reshape(data.shape[0] // feature_num, feature_num)
    #maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
    #归一化 方法：减掉均值，然后除以原取值范围
    #for i in six.moves.range(feature_num-1): data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i]) # six.moves可以兼容python2和python3

    #ratio = 0.8 # 训练集和验证集的划分比例
    #offset = int(data.shape[0]*ratio)
    # train_data = data[:offset]
    # test_data = data[offset:]

    batch_size = 20

    # train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500), batch_size=batch_size)
    train_reader = paddle.batch(paddle.reader.shuffle(train(), buf_size=500), batch_size=batch_size)
    test_reader = paddle.batch(paddle.reader.shuffle(test(), buf_size=500), batch_size=batch_size)


    # feature vector of length 1
    # 定义输入的形状和数据类型
    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32') # 定义输出的形状和数据类型
    y_predict = fluid.layers.fc(input=x, size=1, act=None) # 连接输入和输出的全连接层

    main_program = fluid.default_main_program() # 获取默认/全局主函数
    startup_program = fluid.default_startup_program() # 获取默认/全局启动程序

    cost = fluid.layers.square_error_cost(input=y_predict, label=y) # 利用标签数据和输出的预测数据估计方差
    avg_loss = fluid.layers.mean(cost) # 对方差求均值，得到平均损失

    # SGD optimizer，learning_rate 是学习率，与网络的训练收敛速度有关系
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

#克隆main_program得到test_program
#有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
#该api不会删除任何操作符,请在backward和optimization之前使用
    test_program = main_program.clone(for_test=True)

    # can use CPU or GPU
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Specify the directory to save the parameters
    # 保存训练参数
    params_dirname = "fit_a_line.inference.model"
    num_epochs = 400

    # main train loop. 训练
    # train_test
    # 参数有executor,program,reader,feeder,fetch_list
    # executor表示之前创建的执行器
    # program表示执行器所执行的program
    # reader表示读取到的数据
    # feeder表示前向输入的变量
    # fetch_list表示用户想得到的变量或者命名的结果
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe.run(startup_program)

    train_prompt = "Train cost"
    test_prompt = "Test cost"
    step = 0

    exe_test = fluid.Executor(place)

    # 训练主循环
    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[avg_loss])
            if step % 10 == 0:  # record a train cost every 10 batches每10个批次记录并输出一下训练损失
                print("%s, Step %d, Cost %f" %
                      (train_prompt, step, avg_loss_value[0]))

            if step % 100 == 0:  # record a test cost every 100 batches
                test_metics = train_test(
                    executor=exe_test,
                    program=test_program,
                    reader=test_reader,
                    fetch_list=[avg_loss],
                    feeder=feeder)
                print("%s, Step %d, Cost %f" %
                      (test_prompt, step, test_metics[0]))
                # If the accuracy is good enough, we can stop the training.
                if test_metics[0] < 10.0:
                    break

            step += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")
        if params_dirname is not None:
            # We can save the trained parameters for the inferences later
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict],
                                          exe)
'''
# 预测
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # infer
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets
         ] = fluid.io.load_inference_model(params_dirname, infer_exe)
        batch_size = 10

        infer_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)

        infer_data = next(infer_reader())
        infer_feat = numpy.array(
            [data[0] for data in infer_data]).astype("float32") # 提取测试集中的数据
        infer_label = numpy.array(
            [data[1] for data in infer_data]).astype("float32")  # 提取测试集中的标签


        assert feed_target_names[0] == 'x'
        results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: numpy.array(infer_feat)},
            fetch_list=fetch_targets)  # 进行预测

        print("infer results: (House Price)")
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))

        print("\nground truth:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))

        save_result(results[0], infer_label)
'''

if __name__ == '__main__':
    main()