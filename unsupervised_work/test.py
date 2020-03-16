# -*- encoding: utf-8 -*-
'''
@Author  :   wangjian
'''

import os
from openvino.inference_engine import IENetwork, IECore
import sys
import numpy as np
import time
from tensorflow.contrib import predictor

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class TfModel:
    def __init__(self):
        self._load_model()
    
    def _load_model(self):
        model_dir = os.path.join(CUR_DIR, 'release')
        self.predict_fn = predictor.from_saved_model(model_dir)
    
    def encode(self, input_ids, input_mask, segment_ids):
        return self.predict_fn({'input_ids':input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids})


class VinoModel:
    def __init__(self):
        self._load_model()
    
    def _load_model(self):
        model_dir = os.path.join(CUR_DIR, 'openvino_model')

        model_xml = os.path.join(model_dir,'bert_model.ckpt.xml')
        model_bin = os.path.join(model_dir,'bert_model.ckpt.bin')
        #model_xml = "../test/bert_model.ckpt.xml"
        #model_bin = '../test/bert_model.ckpt.bin'


        device = "CPU"

        ie = IECore()

        print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = IENetwork(model=model_xml, weights=model_bin)

        supported_layers = ie.query_network(net, device)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format('cpu', ', '.join(not_supported_layers)))
            print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
            sys.exit(1)

        print('input_layers:',net.inputs.keys())
        print('output_layers:',net.outputs)

        print("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name=device)
    def encode(self, input_ids, input_mask, segment_ids):
        print("Starting inference")
        inputs = {
            'input_mask' : input_mask,
            'input_ids' : input_ids,
            'segment_ids' : segment_ids
        }
        return self.exec_net.infer(inputs=inputs)


def test_openvino(length=25):
    model = VinoModel()
    input_ids = np.ones([1,length], dtype=np.int)
    input_mask = np.ones([1,length], dtype=np.int)
    segment_ids = np.zeros([1,length], dtype=np.int)

    model.encode(input_ids, input_mask, segment_ids)

    total_time = []
    for i in range(10):
        t1=time.time()
        input_ids = np.random.randint(0,20000,[1,length])
        input_mask = np.ones([1,length],dtype=np.int)
        res = model.encode(input_ids, input_mask, segment_ids)
        # print(res)
        cost_time = time.time()-t1
        total_time.append(cost_time)
        print("{} : {}".format(i,cost_time))
    print("avg time:{}".format(sum(total_time)/len(total_time)))
    return sum(total_time)/len(total_time)

def test_tf(length=25):
    model = TfModel()
    input_ids = np.ones([1,length], dtype=np.int)
    input_mask = np.ones([1,length], dtype=np.int)
    segment_ids = np.zeros([1,length], dtype=np.int)

    model.encode(input_ids, input_mask, segment_ids)

    total_time = []
    for i in range(10):
        t1=time.time()
        input_ids = np.random.randint(0,20000,[1,length])
        input_mask = np.ones([1,length],dtype=np.int)
        res = model.encode(input_ids, input_mask, segment_ids)
        # print(res)
        cost_time = time.time()-t1
        total_time.append(cost_time)
        print("{} : {}".format(i,cost_time))
    print("avg time:{}".format(sum(total_time)/len(total_time)))
    return sum(total_time)/len(total_time)
if __name__=="__main__":
    # 测试openvino
    length=25
    res2=test_tf(length)
    res1=test_openvino(length)
    print((res2-res1)/res2)