#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Will People Like Your Image?

tested with TensorFlow 1.1.0-rc1 (git rev-parse HEAD 45115c0a985815feef3a97a13d6b082997b38e5d) and OpenCV 3.1.0

EXAMPLE:

    python score.py --images "pattern/to/images/*.jpg"
"""
import re

import numpy as np
import tensorflow.compat.v1 as tf
import torch
from torchvision.models import resnet50

if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        new_saver = tf.train.import_meta_graph('../demo/data/ae_model.meta')
        sess.run(tf.global_variables_initializer())
        new_saver.restore(sess, '../demo/data/ae_model')

        feed_node = tf.get_default_graph().get_tensor_by_name("tower_0/image_input:0")
        embedding = tf.get_default_graph().get_tensor_by_name("tower_0/encodings:0")

        model = resnet50()
        model.maxpool.padding = 0
        model.maxpool.ceil_mode = True

        att = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]

        def print_unit(b, u, shortcut=True, shortcut2=False):
            print(f"block {b}: unit {u}")
            units = f"tower_0/resnet_v1_50/block{b}/unit_{u}"

            for t in att:
                if units in t.name:
                    print(t.name)

            c1 = tf.get_default_graph().get_tensor_by_name(f"{units}/bottleneck_v1/conv1/convolution:0")
            c2 = tf.get_default_graph().get_tensor_by_name(f"{units}/bottleneck_v1/conv2/convolution:0")
            c3 = tf.get_default_graph().get_tensor_by_name(f"{units}/bottleneck_v1/conv3/convolution:0")

            print("strides")
            print("c1", c1.op.node_def.attr["strides"])
            print("c2", c2.op.node_def.attr["strides"])
            print("c3", c3.op.node_def.attr["strides"])

            if shortcut2:
                sc = tf.get_default_graph().get_tensor_by_name(f"{units}/bottleneck_v1/shortcut/MaxPool:0")
                print("sc", sc.op.node_def.attr["strides"])
            elif shortcut:
                sc = tf.get_default_graph().get_tensor_by_name(f"{units}/bottleneck_v1/shortcut/convolution:0")
                print("sc", sc.op.node_def.attr["strides"])

        # DEBUG
        # print_unit(3, 1)
        # print_unit(4, 3, shortcut2=True)
        # print(model)
        # exit()

        copy_parameters = True
        if copy_parameters:
            for v in tf.trainable_variables():
                # DEBUG
                # print(v.name, v.shape)
                pass

            i = 0
            for n, p in model.named_parameters():
                pattern = "XXXX"
                squeeze = False

                if n == "conv1.weight":
                    pattern = "resnet_v1_50/conv1/weights"
                elif n == "bn1.weight":
                    pattern = "resnet_v1_50/conv1/BatchNorm/gamma"
                elif n == "bn1.bias":
                    pattern = "resnet_v1_50/conv1/BatchNorm/beta"
                elif n == "fc.weight":
                    pattern = "resnet_v1_50/logits/weights"
                    squeeze = True
                elif n == "fc.bias":
                    pattern = "resnet_v1_50/logits/biases"
                else:
                    block = n[5]
                    unit = int(n[7]) + 1

                    if "downsample" in n:
                        if "0.weight" in n:
                            pattern_wnb = "weights"
                        elif "1.weight" in n:
                            pattern_wnb = "BatchNorm/gamma"
                        elif "1.bias" in n:
                            pattern_wnb = "BatchNorm/beta"
                        else:
                            raise ValueError()

                        pattern_module = f"shortcut/{pattern_wnb}"
                    else:
                        mo = re.match('.+([0-9])[^0-9]*$', n)

                        if "conv" in n:
                            pattern_wnb = "weights"
                        elif "bn" in n and "weight" in n:
                            pattern_wnb = "BatchNorm/gamma"
                        elif "bn" in n and "bias" in n:
                            pattern_wnb = "BatchNorm/beta"
                        else:
                            raise ValueError()

                        pattern_module = f"conv{mo.group(1)}/{pattern_wnb}"

                    pattern = f"block{block}/unit_{unit}/bottleneck_v1/{pattern_module}"

                print(f"looking for {n} : {pattern}")
                for v in tf.trainable_variables():
                    if pattern in v.name:
                        # same shape
                        print(f"   found match: {v.name}")
                        assert ([x for x in p.shape] == list(reversed([x.value for x in v.shape.dims]))) or squeeze

                        value = sess.run(v)

                        if squeeze:
                            value = value.squeeze()

                        if len(value.shape) == 2:
                            value = np.transpose(value)
                        elif len(value.shape) == 4:
                            value = np.transpose(value, (3, 2, 0, 1))

                        new_val = torch.from_numpy(value)

                        p.data = new_val
                        print(f"   assigned tensor to {n} with shape {p.data.shape}")
                        i += 1
                        break

                # DEBUG
                # print(n, p.shape)

            print(f"assigned {i} variables")
            print("copying batchnorm statistics")


            def copy_bn_stats(bn, tensor_name):
                rmt = tf.get_default_graph().get_tensor_by_name(f"{tensor_name}/moving_mean:0")
                rvt = tf.get_default_graph().get_tensor_by_name(f"{tensor_name}/moving_variance:0")
                bn.running_mean.data = torch.tensor(sess.run(rmt))
                bn.running_var.data = torch.tensor(sess.run(rvt))


            copy_bn_stats(model.bn1, "resnet_v1_50/conv1/BatchNorm")

            for block in range(1, 5):
                block_m = getattr(model, f"layer{block}")

                for unit in range(6):
                    if not hasattr(block_m, str(unit)):
                        continue

                    unit_m = getattr(block_m, str(unit))

                    if unit_m.downsample is not None:
                        copy_bn_stats(unit_m.downsample[1],
                                      f"resnet_v1_50/block{block}/unit_{unit + 1}/bottleneck_v1/shortcut/BatchNorm")
                        print(f"layer{block}.{unit}.downsample")

                    for bn_idx in range(1, 4):
                        if not hasattr(unit_m, f"bn{bn_idx}"):
                            continue

                        print(f"layer{block}.{unit}.bn{bn_idx}")
                        copy_bn_stats(getattr(unit_m, f"bn{bn_idx}"),
                                      f"resnet_v1_50/block{block}/unit_{unit + 1}/bottleneck_v1/conv{bn_idx}/BatchNorm")

        # fix conv strides
        model.layer1[2].downsample = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(2, 2), padding=0, ceil_mode=False)
        model.layer1[2].conv2.stride = (2, 2)
        model.layer2[0].conv2.stride = (1, 1)
        model.layer2[0].downsample[0].stride = (1, 1)
        model.layer2[3].conv2.stride = (2, 2)
        model.layer2[3].downsample = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(2, 2), padding=0, ceil_mode=False)
        model.layer3[0].conv2.stride = (1, 1)
        model.layer3[0].downsample[0].stride = (1, 1)
        model.layer3[5].conv2.stride = (2, 2)
        model.layer3[5].downsample = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(2, 2), padding=0, ceil_mode=False)
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
        # end fix

        print(model)
        model.eval()

        model(torch.rand(1, 3, 224, 224))
        torch.save(model, "ae_model_pytorch.pt")
