学习使用openvino， 探索在bert上的效果

# 1. 安装openvino

安装参考 官网 `https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html`

# 2. 模型尝试

## 无监督任务（`./unsupervised_work`）

* 句向量任务
使用bert倒数第二次求和作为句子的编码

1. 将原始的bert模型保存为两种模型（`./unsupervised_work/release_model.py`）

    ```shell
    cd unsupervised_work
    python3.5 release_model.py
    ```

    生成两种模型
    * `./unsupervised_work/bert_model.ckpt.pb`，用于openvino生成IR模型
    * `./unsupervised_work/release`，用于tensorflow的inference

2. openvino 生成IR模型
    在openvino的安装目录下的model_optimizer文件夹下执行
    我的目录,默认安装目录（/opt/intel/openvino/deployment_tools/model_optimizer）

    ```
    cd /opt/intel/openvino/deployment_tools/model_optimizer
    python3.5 ./mo_tf.py --input_model  /home/wangjian0110/myWork/learning/learning_openvino/unsupervised_work/bert_model.ckpt.pb --disable_nhwc_to_nchw    --output_dir /home/wangjian0110/myWork/learning/learning_openvino/unsupervised_work/openvino_model 
    ```

3. 测试(`./unsupervised_work/test.py`)

```shell
cd unsupervised_work
python3.5 test.py
```

| 序号 | 模型 | 时间  | 占用内存 | 
| :-: | :-: | :-: | :-: | 
| 1 | openvino | 0.13 | 40% | 
| 2 |  tf | 0.1 | 10%|

4. 结论

* 长度为25的情况下，openvino时间可提升20%-40%（多次实验，基本在这个范围之内）
* openvino可节省内存约70%（个人评估，不准确）

5. 模型（百度网盘）

链接: https://pan.baidu.com/s/1Sr6-tMyN041SigXxhoyMFw 提取码: nqm9
