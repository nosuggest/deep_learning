# 安装
- docker
- tensorflow/serving:latest-devel
    - docker pull tensorflow/serving:latest-devel

# 模型导出
- 输入序列化后的tf.train.Example
    - tf.estimator.export.build_parsing_serving_input_receiver_fn
    - tf.feature_column.make_parse_example_spec
- 输入 TensorProto 字典
    - tf.estimator.export.build_raw_serving_input_receiver_fn
    - tf.placeholder
    
# 模型serving
- docker部署
    - docker pull tensorflow/serving:latest-devel #拉取
    - docker run -it -p 8500:8500 tensorflow/serving:latest-devel #启动  
    - docker ps  # 查看container id
    - docker cp $TESTDATA/saved_model  container_id:/online_model
    - tensorflow_model_server --port=8500 --rest_api_port=8500 --model_name=textcnn_model --model_base_path=/online_model
    
# 请求
- predict_pb2,prediction_service_pb2_grpc请求
- 注意，host为docker部署的host
- 代码：Estimator框架模版/Serving/serving_grpc_client.py


# 结果
'''
outputs {
  key: "prob"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 14
      }
    }
    float_val: 0.0007316882838495076
    float_val: 0.00012417411198839545
    float_val: -0.001078148139640689
    float_val: 0.000776230008341372
    float_val: 0.0004840499022975564
    float_val: 0.001928473706357181
    float_val: -0.0002663416089490056
    float_val: 0.0003809597692452371
    float_val: -0.0012025543255731463
    float_val: 0.0019091889262199402
    float_val: 0.00048460805555805564
    float_val: 1.0264695447403938e-05
    float_val: -0.0007654627552255988
    float_val: -0.0002853185869753361
  }
}
model_spec {
  name: "textcnn_model"
  version {
    value: 1578464652
  }
  signature_name: "serving_default"
}
'''