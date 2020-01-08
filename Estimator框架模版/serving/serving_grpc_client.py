#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/8 3:55 PM
# @Author  : Slade
# @File    : serving_grpc_client.py

import numpy as np
import requests
import json
import base64
import time

from tensorflow_serving.apis import predict_pb2,prediction_service_pb2_grpc

import grpc
import tensorflow as tf
import random

host = '127.0.0.1'
data = {}
data['sentence'] = [random.randint(0, 100) for i in range(55)]


def grpc_predict_raw(data):
    port = 8500
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    # channel = implementations.insecure_channel(host, int(port))

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'textcnn_model'
    request.model_spec.signature_name = "serving_default"

    tensor_protos = {
        # 一条一条的请求方式
        'sentence':tf.make_tensor_proto(data['sentence'], dtype=tf.int64, shape=[1, 55])
    }
    for k in tensor_protos:
        request.inputs[k].CopyFrom(tensor_protos[k])

    response = stub.Predict(request, 5.0)
    print(response)

if __name__ == '__main__':
    grpc_predict_raw(data)