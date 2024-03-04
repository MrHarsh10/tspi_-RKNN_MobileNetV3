import cv2
import numpy as np
from rknnlite.api import RKNNLite
INPUT_SIZE = 224
RK3566_MODEL = 'tspi_moblienetv3_demo.rknn'
labels=["cat","cattle","dog","house","pig"]
def show_top5(result):
    output = result[0].reshape(-1)
    labels=["cat","cattle","dog","house","pig"]
    # Softmax
    output = np.exp(output) / np.sum(np.exp(output))
    # Get the indices of the top 5 largest values
    output_sorted_indices = np.argsort(output)[::-1][:5]
    top5_str = '-----TOP 5-----\n'
    for i, index in enumerate(output_sorted_indices):
        value = output[index]
        if value > 0:
            topi = '[{:>3d}] score:{:.6f} class:"{}"\n'.format(index, value, labels[index])
        else:
            topi = '-1: 0.0\n'
        top5_str += topi
    print(top5_str)


if __name__ == '__main__':



    rknn_lite = RKNNLite()
    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(RK3566_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')


    ori_img = cv2.imread('./demo_cat.jpg')
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[img])

    # Show the classification results
    show_top5(outputs)
    print('done')

    rknn_lite.release()