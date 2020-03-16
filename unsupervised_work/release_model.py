# -*- encoding: utf-8 -*-
'''
@Author  :   wangjian
'''

import os
import tensorflow as tf
from tensorflow.python.framework import graph_io
import shutil
import sys
# 加载bert模型，并且输出导出第二层

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from bert_tools.modeling import BertConfig, BertModel


class Config:
    # 基本配置
    pretrain_model_dir = '/home/wangjian0110/myWork/chinese_roberta_wwm_ext_L-12_H-768_A-12'  # 可从 https://github.com/ymcui/Chinese-BERT-wwm 下载
    #pretrain_model_dir = '/home/wangjian0110/myWork/chinese_L-12_H-768_A-12_2'  # google官方
    bert_model_file = os.path.join(pretrain_model_dir, 'bert_model.ckpt')
    release_folder = os.path.join(CUR_DIR, 'release')
    save_folder = os.path.join(CUR_DIR, 'save')
    bert_config_file = os.path.join(pretrain_model_dir, 'bert_config.json')
    batch_size = 1
    max_seq_length = 25


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    config = Config()

    release_folders = [config.release_folder, config.save_folder]
    for release_folder in release_folders:
        if os.path.isdir(release_folder):
            tf.logging.info(
                "release folder already exists, and isn't empty. Will delete the folder."
            )
            shutil.rmtree(release_folder)
        #tf.gfile.MakeDirs(release_folder)

    bert_config = BertConfig.from_json_file(config.bert_config_file)

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
    masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
        tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

    batch_size = config.batch_size
    max_seq_length = config.max_seq_length

    g = tf.Graph()
    with g.as_default():
        with tf.device(None):
            with tf.Session(config=session_config) as sess:
                input_ids = tf.placeholder(dtype=tf.int64,
                                        shape=[batch_size, max_seq_length],
                                        name='input_ids')
                input_mask = tf.placeholder(dtype=tf.int64,
                                            shape=[batch_size, max_seq_length],
                                            name='input_mask')
                segmet_ids = tf.placeholder(dtype=tf.int64,
                                            shape=[batch_size, max_seq_length],
                                            name='segment_ids')
                model = BertModel(config=bert_config,
                                is_training=False,
                                input_ids=input_ids,
                                input_mask=input_mask,
                                token_type_ids=segmet_ids)
                # print(input_ids)
                # print(input_mask)
                input_mask_float = tf.cast(input_mask, tf.float32)
                encoder_layer = model.get_all_encoder_layers()[-2]
                pooled = masked_reduce_mean(encoder_layer, input_mask_float)
                saver = tf.train.Saver()
                saver.restore(sess, config.bert_model_file)
                #saver.save(sess, os.path.join(config.save_folder,
                #                              'bert_model.ckpt'))
                print(pooled.name)
                #exit()
                # print(sess.graph_def)
                # print(model.pooled_output.name)
                frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['truediv'])
                graph_io.write_graph(frozen, './', 'bert_model.ckpt.pb', as_text=False)
                tf.saved_model.simple_save(session=sess,
                                         export_dir=config.release_folder,
                                         inputs={
                                             'input_ids': input_ids,
                                             'input_mask': input_mask,
                                             'segment_ids':segmet_ids
                                         },
                                         outputs={'encode': pooled})
    # print(pooled)


if __name__ == "__main__":
    main()
