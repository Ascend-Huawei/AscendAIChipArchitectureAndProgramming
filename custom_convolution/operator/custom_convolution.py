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
import math
import numpy as np
import te.lang.cce
from te import tvm
from topi import generic
from te.platform.cce_buffer import cur_cce_product_params as cce_product
from te import platform as cceconf
from topi.cce import util
from te.platform import CUBE_MKN
import os


# the dim of shape in conv must be 4
CONV_SHAPE_DIM = 4

# padH, padW must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

# stride must be in [1,64]
STRIDE_MIN = 1
STRIDE_MAX = 63

# sizeof(fp16) = 2, sizeof(8bit) = 1
SIZE_OF_FP16 = 2
SIZE_OF_8BIT = 1

NoneType = type(None)


def check_pad_and_return_avail_m(shape_in, shape_w, in_dtype, w_dtype, padu, padl, strideH, strideW):
    mBitLength = {"float32":32,"float16":16,"uint8":8,"int8":8, "uint4":4,"int4":4}
    mBitRatio = {"int32":4,"float32":4,"float16":2,"uint8":1,"int8":1, "uint4":1.0/2,"int4":1.0/2}
    config = CUBE_MKN[w_dtype]
    ci0 = config['mac'][1]
    hi = shape_in[2]
    wi = shape_in[3]
    hk = shape_w[2]
    wk = shape_w[3]

    ho = (hi + (2 * padu) - hk) / strideH + 1; # calculated by hi and wi
    wo = (wi + (2 * padl) - wk) / strideW + 1; # calculated by hi and wi

    m_max = min(cce_product.getParams("L0A_Buffer")/mBitRatio[in_dtype]/16/ci0,
        cce_product.getParams("L0C_Buffer")/mBitRatio["float16"]/16/16)
    M = ho * wo
    m = [i for i in np.arange(1, min(m_max, math.ceil(float(M)/16)) + 1).astype(int)]

    m_common_factor = math.ceil(float(M)/16)
    m_selected = []
    AL_Hi = []
    m_avail = []
    for p_m in m:
        if p_m in compute_common_factor(m_common_factor):
            m_selected.append(p_m)

    for m in m_selected:
        AL_Wo = wo
        AL_Ho = math.ceil(float(m*16)/AL_Wo) + 1
        tmp = (AL_Ho-1)*strideH + hk
        if tmp >= padu + 1:
            AL_Hi.append(tmp)
            m_avail.append(m)

    return AL_Hi, m_avail

def compute_common_factor(num):
    if num == 0: return [0]
    if num == 1: return [1]
    rlist = []
    i = 1
    while i <= num:
        if num % i == 0:
            rlist.append(i)
        i += 1
    rlist.sort(reverse = True) # descending sort
    return rlist

def check_tail_without_data(m_input, wo, shape_in, shape_w, strideh, padu, mBitLength):
    m_selected = []
    for m in m_input:
        split_row_in_H = math.ceil(float(m * mBitLength['float16']) / wo) * strideh + shape_w[2]
        if strideh <= shape_w[2]:
            tmp = (shape_in[2] + 2 * padu) % (split_row_in_H - (shape_w[2] - strideh))
            if tmp == 0 or tmp > padu:
                m_selected.append(m)
        else:
             tmp = (shape_in[2] + 2 * padu) % (split_row_in_H + (strideh - shape_w[2]))
             if tmp == 0 or tmp > padu:
                m_selected.append(m)
    if m_selected == []:
        raise RuntimeError("Tail data contains no data, the input shape can not fullfil!")
    else:
        return m_selected

def check_CUB_overflow(m_selected, res_dtype, mBitLength, mBitRatio):
    ubBufferSize = 0.25 * cce_product.getParams("Unified_Buffer") / mBitRatio['float16']
    if (res_dtype == "uint8" or res_dtype == "int8"):
        nPart = 2
    else:
        nPart = 1
    if min(m_selected) * nPart * mBitLength['float16'] * mBitLength['float16'] >= ubBufferSize:
        raise RuntimeError("CUB size overflow UB buffer!")

def conv_check_rule(shape_in, shape_w, in_dtype, w_dtype, padh, padw, strideh, stridew):

    padl, padr, padu, padd = padw, padw, padh, padh
    batch = shape_in[0]
    hi = shape_in[2]
    wi = shape_in[3]

    config = CUBE_MKN[w_dtype]
    ci0 = config['mac'][1]
    ci1 = ((shape_in[1]) + ci0 - 1) // ci0

    co0 = 16 # The unit of channel is 16
    co1 = (shape_w[0] + co0 -1) / co0
    hk = shape_w[2]
    wk = shape_w[3]

    # ============ conv case checking begin ================
    mBitLength = {"float32":32, "float16":16, "uint8":8, "int8":8, "uint4":4, "int4":4}
    mBitRatio = {"int32":4, "float32":4, "float16":2, "uint8":1, "int8":1, "uint4":1.0/2, "int4":1.0/2}
    inputDataType = in_dtype # "uint8" if img_dtype == 0 else "int8" if img_dtype == 1 else "float16"

    # added for checking pad
    # avoid for no real data in feature map when load3d
    m_target = 1
    wo = (wi + (2 * padl) - wk) / stridew + 1
    tmp1 = ((m_target * mBitLength['float16']) + wo - 1) / wo
    tmp2 = ((tmp1 * strideh) + hk) * (wi + (2 * padl))
    MaxFeatureMap = 1 * ci0 * tmp2 * 2 * mBitRatio[inputDataType]

    L1BufferSize = cce_product.getParams("L1_Buffer")  # bytes

    if MaxFeatureMap > L1BufferSize:
        raise RuntimeError("L1 buffer overflow!")

    ho = (hi + (2 * padu) - hk) / strideh + 1

    if np.int64(batch * wo * ho * shape_w[0]) >= np.int64(2**31)-1:
        raise RuntimeError("Output fmap exceed 32 bit limitations!")

    if np.int64(batch*hi*wi*ci1*ci0) >= np.int64(2**31)-1:
        raise RuntimeError("Input fmap exceed 32 bit limitations!")


@util.check_input_type((list, tuple), (list, tuple), str, str, str, int, int, int, int, int,
                       str, int, int)
def conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype, padh, padw, strideh, stridew,  bias=0,
                   kernel_name="conv_layer_cce", need_build=0, need_print=0):
    """

    Parameters
    ----------
    shape_in : shape of data_in

    shape_w : shape of filter

    in_dtype : the feature map data type

    w_dtype : the weight data type

    res_dtype : the result data type

    padh: the padding shape in H

    padw: the padding shape in Weight

    strideh: the stride value in H

    stridew: the stride value in Weight

    quantizeConfig: quantize config table, default [0, 0, 0]
    quantizeConfig[0] - quantize function switch
                        0: quantize off
                        1: quantize on
    quantizeConfig[1] - QuantizeAlgorithm
                        0: non offset
                        1: half offset
                        2: all offset ( Not supported now )
    quantizeConfig[2] - QuantizeScaleType (for Dequantize/Requantize, quantize always scalar)
                        0: scalar
                        1: vector

    scaleSqrt: scale mode
    scaleSqrt[0] - Quantize scale mode
                   0: non sqrt
                   1: sqrt
    scaleSqrt[1] - DeQuantize scale mode
                   0: non sqrt
                   1: sqrt
    scaleSqrt[2] - ReQuantize scale mode
                   0: non sqrt
                   1: sqrt

    scaleQ_dtype: Quantize scale data type, default 'float16'

    offsetQ_dtype: Quantize offset data type, default 'float16'

    scaleDq_dtype: DeQuantize scale data type, default 'float16'

    scaleRq_dtype: ReQuantize scale data type, default 'float16'

    offsetRq_dtype: ReQuantize offset data type, default 'float16'

    offsetW_dtype: Weight offset data type, default 'int32'

    offsetPad_dtype: Quantize Cube offset data type, default 'uint8'

    bias: the tag for bias or not

    kernel_name : cce kernel name, default value is "cce_conv"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None

    """
    # for pylint, otherwise "Dangerous default value [] as argument"
#    if quantizeConfig is None:
#        quantizeConfig = [0, 0, 0]
#    if scaleSqrt is None:
#        scaleSqrt = [0, 0, 0]

    # conv shape check
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_in, CONV_SHAPE_DIM, CONV_SHAPE_DIM)
    util.check_shape_rule(shape_w, CONV_SHAPE_DIM, CONV_SHAPE_DIM)

    in_dtype = in_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()
#    scaleQ_dtype = scaleQ_dtype.lower()
#    offsetQ_dtype = offsetQ_dtype.lower()
#    scaleDq_dtype = scaleDq_dtype.lower()
#    scaleRq_dtype = scaleRq_dtype.lower()
#    offsetRq_dtype = offsetRq_dtype.lower()
#    offsetW_dtype = offsetW_dtype.lower()
#    offsetPad_dtype = offsetPad_dtype.lower()

    # conv data type check
    util.check_dtype_rule(in_dtype, ['float16', 'int8', 'uint8'])
    util.check_dtype_rule(w_dtype, ['float16', 'int8', 'uint8'])
    util.check_dtype_rule(res_dtype, ['float16', 'int8', 'uint8'])
#    util.check_dtype_rule(scaleQ_dtype, ['float16'])
#    util.check_dtype_rule(offsetQ_dtype, ['float16'])
#    util.check_dtype_rule(scaleDq_dtype, ['float16'])
#    util.check_dtype_rule(scaleRq_dtype, ['float16'])
#    util.check_dtype_rule(offsetRq_dtype, ['float16'])
#    util.check_dtype_rule(offsetW_dtype, ['int32'])
#    util.check_dtype_rule(offsetPad_dtype, ['uint8'])

#    if quantizeConfig[0] == 0:
    util.check_dtype_rule(in_dtype, ['float16'])
    util.check_dtype_rule(w_dtype, ['float16'])
    util.check_dtype_rule(res_dtype, ['float16'])

#    if quantizeConfig[0] == 1:
#        util.check_dtype_rule(w_dtype, ['int8'])

    shape_in=list(shape_in)
    shape_w=list(shape_w)

#    shape_in, shape_w = te.lang.cce.check_conv_shape(shape_in, shape_w, padh, padw, strideh,
#                                                     stridew, in_dtype, w_dtype, res_dtype)

#    if shape_in[1]!=shape_w[1]:
#        raise RuntimeError("shape_in[1] must equal to shape_w[1]")

    block_size_K = CUBE_MKN[in_dtype]['mac'][1]
    shape_in[1]=((shape_in[1]+block_size_K-1)//block_size_K)*block_size_K
    shape_w[1]=shape_in[1]

    hi = shape_in[2]
    wi = shape_in[3]
    hk = shape_w[2]
    wk = shape_w[3]
    h_out = 0
    w_out = 0
#    print(hi)
#    print(wi)
#    print(hk)
#    print(wk)
 #   print(strideh)
#    print(stridew)
#    print(padh)
#    print(padw)
    if strideh != 0:
        h_out = (hi + (2 * padh) - hk) / strideh + 1 # calculated by hi and wi
    if stridew != 0:
        w_out = (wi + (2 * padw) - wk) / stridew + 1 # calculated by hi and wi

    if h_out<=0:
        raise RuntimeError("h_out must >0, h_out = (hi + (2 * padh) - hk) / strideh + 1")
    if w_out<=0:
        raise RuntimeError("w_out must >0, w_out = (wi + (2 * padw) - wk) / stridew + 1")

    if padh > hk:
        raise RuntimeError("kernel H must >= Pad H")

    if (shape_in[0]*w_out*h_out*hk*wk*CUBE_MKN[w_dtype]['mac'][1]) > (np.int64(2**31)-1):
        raise RuntimeError("im2col shape exceed 32bit limitation")

    conv_check_rule(shape_in, shape_w, in_dtype, w_dtype, padh, padw, strideh, stridew)

    if res_dtype in ['int8', 'uint8']:
        w_block_size_K = CUBE_MKN[w_dtype]['mac'][1]
        shape_w[0] = ((shape_w[0]+w_block_size_K-1)//w_block_size_K)*w_block_size_K
    else:
        w_block_size_N = CUBE_MKN[w_dtype]['mac'][2]
        shape_w[0] = ((shape_w[0]+w_block_size_N-1)//w_block_size_N)*w_block_size_N

    # padh, padw check
    if padh < PAD_MIN or padh > PAD_MAX:
        raise RuntimeError("padh must be in [0,255].")
    if padw < PAD_MIN or padw > PAD_MAX:
        raise RuntimeError("padw must be in [0,255].")

    # strideh, stridew check
    if strideh < STRIDE_MIN or strideh > STRIDE_MAX:
        raise RuntimeError("strideh must be in [1,63].")
    if stridew < STRIDE_MIN or stridew > STRIDE_MAX:
        raise RuntimeError("stridew must be in [1,63].")

    # filterH, filterW check
    if shape_w[2] < FILTER_HW_MIN or shape_w[2] > FILTER_HW_MAX:
        raise RuntimeError("filterh must be in [1,255].")
    if shape_w[3] < FILTER_HW_MIN or shape_w[3] > FILTER_HW_MAX:
        raise RuntimeError("filterw must be in [1,255].")

    # tiling check, filterH*inputC*inputW*sizeof(in_dtype) < half of(L1_BUFFER)
    SIZE_OF_L1_BUFFER = cce_product.getParams("L1_Buffer")  # bytes

    if (in_dtype == 'float16'):
        if (shape_w[2]) * (shape_in[1]) * (shape_in[3]) * SIZE_OF_FP16 > (SIZE_OF_L1_BUFFER / 2):
            raise RuntimeError("min cut is out of half of L1 memory.")

    if (in_dtype == 'int8' or in_dtype == 'uint8'):
        if (shape_w[2]) * (shape_in[1]) * (shape_in[3]) * SIZE_OF_8BIT > (SIZE_OF_L1_BUFFER / 2):
            raise RuntimeError("min cut is out of half of L1 memory.")

    # quantize switch on

#    if quantizeConfig[0] == 1:
#        quantizeTurnOn = True
        # quantize -> DeQuantize dataflow
#        if (in_dtype == 'float16' and w_dtype == 'int8' and res_dtype == 'float16'):
#            isQuantize = True
#            isDeQuantize = True
#            isReQuantize = False
        # DeQuantize dataflow
#        elif ((in_dtype == 'int8' or in_dtype == 'uint8') and w_dtype == 'int8' and res_dtype == 'float16'):
#            isQuantize = False
#            isDeQuantize = True
#            isReQuantize = False
        # quantize -> ReQuantize dataflow
#        elif (in_dtype == 'float16' and w_dtype == 'int8' and (res_dtype == 'int8' or res_dtype == 'uint8')):
#            isQuantize = True
#            isDeQuantize = False
#            isReQuantize = True
        # ReQuantize dataflow
#        elif ((in_dtype == 'int8' or in_dtype == 'uint8') and w_dtype == 'int8' and (res_dtype == 'int8' or res_dtype == 'uint8')):
#            isQuantize = False
#            isDeQuantize = False
#            isReQuantize = True
#        else:
#            raise RuntimeError("Not support in/out data type for quantize.")
    # quantize switch off
#    elif quantizeConfig[0] == 0:
    quantizeTurnOn = False
    isQuantize = False
    isDeQuantize = False
    isReQuantize = False
#    else:
#        raise RuntimeError("Invalid Quantize Config.")

    # - - - # - - - # - - - - - - - # - - - - - - # - - - # - - - # - - - - #
    # 07    | 06    | 05      04    | 03          | 02    | 01    | 00      #
    # QSqrt | scale | offset        | ReQ         | DeQ   | Quan  | Switch  #
    # - - - # - - - # - - - # - - - # - - - - - - # - - - # - - - # - - - - #
    # 15    | 14    | 13    | 12    | 11          | 10    | 09    | 08      #
    # Null  | Null  | Null  | Null  |in_dsl_flag  | bias  | RqSqrt| DqSqrt  #
    # - - - # - - - # - - - # - - - # - - - # - - - # - - - # - - - - #
    # in_dsl_flag     #0: to imply conv by ir directly, it's not perferred
    #                 #1: to imply  conv by dsl, it's default way
#    in_dsl_flag = 1  # 0 for old conv
#    te.lang.cce.conv_param.tiling = tiling

    model_config = (1 if quantizeTurnOn else 0)     \
        | (1 if isQuantize else 0) << 1    \
        | (1 if isDeQuantize else 0) << 2  \
        | (1 if isReQuantize else 0) << 3  \
        | 0 << 4           \
        | 0 << 6           \
        | 0 << 7                \
        | 0 << 8                \
        | 0 << 9                \
        | (1 if bias else 0) << 10         \
        | 1 << 11

    with tvm.target.cce():
        Data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype)
        Weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype)

        # bias or fusion_bias(half offset)
        if bias or (model_config & 0x31 == 0x11):
            Bias = tvm.placeholder(
                (shape_w[0], ), name='Bias', dtype="int32" if quantizeTurnOn else "float16")
        # bias or fusion_bias(all offset)
        elif bias or (model_config & 0x31 == 0x21):
            Bias = tvm.placeholder(
                (shape_w[0], ), name='Bias', dtype="uint32" if quantizeTurnOn else "float16")

        # quantize on
        if quantizeTurnOn:
            QuantizeAlgorithm = quantizeConfig[1]
            if isQuantize:
                scaleQ = tvm.placeholder(
                    (CUBE_MKN[scaleQ_dtype]['mac'][1], ), name='scaleQ', dtype=scaleQ_dtype)
                if QuantizeAlgorithm == 1 or QuantizeAlgorithm == 2:
                    offsetQ = tvm.placeholder(
                        (CUBE_MKN[offsetQ_dtype]['mac'][1], ), name='offsetQ', dtype=offsetQ_dtype)

            if isDeQuantize:
                scaleDq_shape = (CUBE_MKN[scaleDq_dtype]['mac'][1], ) if quantizeConfig[
                    2] == 0 else (shape_w[0], )
                scaleDq = tvm.placeholder(
                    scaleDq_shape, name='scaleDq', dtype=scaleDq_dtype)

            if isReQuantize:
                scaleRq_shape = (CUBE_MKN[scaleRq_dtype]['mac'][1], ) if quantizeConfig[
                    2] == 0 else (shape_w[0], )
                scaleRq = tvm.placeholder(
                    scaleRq_shape, name='scaleRq', dtype=scaleRq_dtype)
                if QuantizeAlgorithm == 1 or QuantizeAlgorithm == 2:
                    offsetRq_shape = (CUBE_MKN[offsetRq_dtype]['mac'][1], ) if quantizeConfig[
                        2] == 0 else (shape_w[0], )
                    offsetRq = tvm.placeholder(
                        offsetRq_shape, name='offsetRq', dtype=offsetRq_dtype)
            # need offsetPad , for half offset and all offset
            if QuantizeAlgorithm == 1 or QuantizeAlgorithm == 2:
                offsetPad = tvm.placeholder(
                    (CUBE_MKN[offsetPad_dtype]['mac'][1], ), name='offsetPad', dtype=offsetPad_dtype)

            # non offset
            if QuantizeAlgorithm == 0:
                if bias:
                    if isQuantize:
                        if isDeQuantize:
                            tensor_list = te.lang.cce.conv(
                                Data, Weight, Bias, scaleQ, scaleDq, res_dtype, padh, padw, strideh, stridew, model_config)
                        else:
                            tensor_list = te.lang.cce.conv(
                                Data, Weight, Bias, scaleQ, scaleRq, res_dtype, padh, padw, strideh, stridew, model_config)

                    else:
                        if isDeQuantize:
                            tensor_list = te.lang.cce.conv(
                                Data, Weight, Bias, scaleDq, res_dtype, padh, padw, strideh, stridew, model_config)
                        else:
                            tensor_list = te.lang.cce.conv(
                                Data, Weight, Bias, scaleRq, res_dtype, padh, padw, strideh, stridew, model_config)

                else:
                    if isQuantize:
                        if isDeQuantize:
                            tensor_list = te.lang.cce.conv(
                                Data, Weight, scaleQ, scaleDq, res_dtype, padh, padw, strideh, stridew, model_config)
                        else:
                            tensor_list = te.lang.cce.conv(
                                Data, Weight, scaleQ, scaleRq, res_dtype, padh, padw, strideh, stridew, model_config)

                    else:
                        if isDeQuantize:
                            tensor_list = te.lang.cce.conv(
                                Data, Weight, scaleDq, res_dtype, padh, padw, strideh, stridew, model_config)
                        else:
                            tensor_list = te.lang.cce.conv(
                                Data, Weight, scaleRq, res_dtype, padh, padw, strideh, stridew, model_config)

            # half offset
            elif QuantizeAlgorithm == 1:
                if isQuantize:
                    if isDeQuantize:
                        tensor_list = te.lang.cce.conv(
                            Data, Weight, Bias, scaleQ, offsetQ, scaleDq, offsetPad, res_dtype, padh, padw, strideh, stridew, model_config)
                    else:
                        tensor_list = te.lang.cce.conv(
                            Data, Weight, Bias, scaleQ, offsetQ, scaleRq, offsetRq, offsetPad, res_dtype, padh, padw, strideh, stridew, model_config)

                else:
                    if isDeQuantize:
                        tensor_list = te.lang.cce.conv(
                            Data, Weight, Bias, scaleDq, offsetPad, res_dtype, padh, padw, strideh, stridew, model_config)
                    else:
                        tensor_list = te.lang.cce.conv(
                            Data, Weight, Bias, scaleRq, offsetRq, offsetPad, res_dtype, padh, padw, strideh, stridew, model_config)

            # all offset
            elif QuantizeAlgorithm == 2:
                raise RuntimeError("All Offset mode quantize not support.")
            else:
                raise RuntimeError("Invalid quantize algorithm.")
        # quantize off
        else:
            if bias:
                # Res = Data * Weight + Bias
                tensor_list = te.lang.cce.conv(
                    Data, Weight, Bias, res_dtype, padh, padw, strideh, stridew, model_config)
            else:
                # Res = Data * Weight
                tensor_list = te.lang.cce.conv(
                    Data, Weight, res_dtype, padh, padw, strideh, stridew, model_config)

        tensor_list = list(tensor_list)
        sch = generic.auto_schedule(tensor_list[-1])

    config = {
        "print_ir": need_print,
        "need_build": need_build,
        "name": kernel_name,
        "tensor_list": tensor_list
    }

    te.lang.cce.cce_build_code(sch, config)

if __name__ == "__main__":
    conv_layer_cce((1, 3,64,64),(1,3,3,3), "float16", "float16", "float16",0,0,1,1)