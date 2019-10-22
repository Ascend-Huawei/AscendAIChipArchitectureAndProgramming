"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
import numpy as np


def dump_data(input_data, name, fmt, data_type):
    if fmt == "binary" or fmt == "bin":
        f_output = open(name, "wb")
        if (data_type == "float16"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.float16(elem).tobytes())
        elif (data_type == "float32"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.float32(elem).tobytes())
        elif (data_type == "int32"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.int32(elem).tobytes())
        elif (data_type == "int8"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.int8(elem).tobytes())
        elif (data_type == "uint8"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.uint8(elem).tobytes())
    else:
        f_output = open(name, "w")
        index = 0
        for elem in np.nditer(input_data):
            f_output.write("%f\t" % elem)
            index += 1
            if index % 16 == 0:
                f_output.write("\n")


def gen_reduction_data(name, op, axis, coeff):
    sys.stdout.write("Info: writing input for %s...\n" % name);
    input_shape = (2, 3, 4)

    s_type = np.float16

    input_arr = np.ones(input_shape, dtype=s_type) * -1

    dump_data(input_arr, name + "_input_2_3_4_" + op.lower() + "_axis_" + str(
        axis) + ".txt", fmt="float", data_type="float16")
    dump_data(input_arr, name + "_input_2_3_4_" + op.lower() + "_axis_" + str(
        axis) + ".data", fmt="binary", data_type="float16")

    dims = len(input_shape)
    axis_a = int((dims + axis) % dims)
    new_shape = []
    for i in range(axis_a):
        new_shape.append(input_shape[i])
    new_shape.append(-1)
    print(new_shape)
    input_data = input_arr.reshape(new_shape)
    output_data = None

    if op == "ASUM":
        output_data_tmp = np.add.reduce(np.abs(input_data, dtype=s_type),
                                        axis=-1, dtype=s_type)
    elif op == "SUMSQ":
        output_data_tmp = np.add.reduce(np.square(input_data, dtype=s_type),
                                        axis=-1, dtype=s_type)
    elif op == "MEAN":
        output_data_tmp = np.mean(input_data, axis=-1, dtype=s_type)
    elif op == "SUM":
        output_data_tmp = np.add.reduce(input_data, axis=-1, dtype=s_type)
    else:
        raise RuntimeError("unsupported op:%s " % op)

    output_data = (output_data_tmp * coeff).astype(s_type)

    dump_data(output_data,
              name + "_output_2_3_4_" + op.lower() + "_axis_" + str(
                  axis) + ".txt", fmt="float", data_type="float16")
    dump_data(output_data,
              name + "_output_2_3_4_" + op.lower() + "_axis_" + str(
                  axis) + ".data", fmt="binary", data_type="float16")

    sys.stdout.write("Info: writing output for %s done!!!\n" % name)


def sign(name):
    sys.stdout.write("Info: writing input for sign(2, 4)...\n")
    input_arr_shape = (2, 4)
    input_arr = np.random.uniform(0, 1, size=input_arr_shape).astype(np.float16)
    dump_data(input_arr, name + "_input_2_4_cce.data", fmt="binary",
              data_type="float16")
    dump_data(input_arr, name + "_input_2_4_cce.txt", fmt="float",
              data_type="float16")

    output_arr = np.sign(input_arr)
    dump_data(output_arr, name + "_output_2_4_cce.data", fmt="binary",
              data_type="float16")
    dump_data(output_arr, name + "_output_2_4_cce.txt", fmt="float",
              data_type="float16")


if __name__ == "__main__":
    sign("sign")
    gen_reduction_data("Reduction", "SUM", 1, 2)
