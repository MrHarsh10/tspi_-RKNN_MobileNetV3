import numpy as np
import numpy as np
import cv2
from rknn.api import RKNN
import os
import torch

import os




def show_outputs(output):
    index = sorted(range(len(output)), key=lambda k : output[k], reverse=True)
    fp = open('./labels.txt', 'r')
    labels = fp.readlines()
    top5_str = 'mobilenetv3\n-----TOP 5-----\n'
    for i in range(5):
        value = output[index[i]]
        if value > 0:
            topi = '[{:>3d}] score:{:.6f} class:"{}"\n'.format(index[i], value, labels[index[i]].strip().split(':')[-1])
        else:
            topi = '[ -1]: 0.0\n'
        top5_str += topi
    print(top5_str.strip())


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':


    rknn = RKNN(verbose=True)

    # Pre-process config
    input_size_list = [[1, 3, 224, 224]]
    print('--> Config model')
    rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.395, 58.395, 58.395], target_platform='rk3566')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_pytorch(model="./MobileNetV3.pt", input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('./tspi_moblienetv3_demo.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    img = cv2.imread('./demo_pig.jpg')
    
    # 调整图片大小为模型所需大小
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img], data_format=['nhwc'])
    print(outputs)
    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()