/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the 
License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <Python.h>
#include "custom/custom_op.h"
#include "framework/omg/register.h"
#include "framework/omg/omg_types.h"
#include "proto/caffe/caffe.pb.h"
#include "operator.h"
#include "attr_value.h"
#include <memory>
#include <string>
#include <vector>
using namespace ge;

namespace domi
{
// Caffe ParseParams function
Status CaffeReductionParseParams(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_origin to layer
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    // #### Verify the validity of input operator parameters.
    if (nullptr == layer)
    {
        printf("Dynamic cast op_src to LayerParameter failed\n");
        return FAILED;
    }
    // #### Map corresponding to the operator operation type and its character string
    std::map<caffe::ReductionParameter_ReductionOp, std::string> operation_map = {
        { caffe::ReductionParameter_ReductionOp_SUM, "SUM" },
    { caffe::ReductionParameter_ReductionOp_ASUM, "ASUM" },
    { caffe::ReductionParameter_ReductionOp_SUMSQ, "SUMSQ" },
    { caffe::ReductionParameter_ReductionOp_MEAN, "MEAN" },
    };
    // #### Obtains operator parameters.
    const caffe::ReductionParameter& param = layer->reduction_param();
    if(param.has_axis())
    {
        op_dest.SetAttr("axis", AttrValue::CreateFrom<AttrValue::INT>(param.axis()));
    }
    if(param.has_coeff())
    {
        op_dest.SetAttr("coeff", AttrValue::CreateFrom<AttrValue::FLOAT>(param.coeff()));
    }
    if(param.has_operation())
    {
        op_dest.SetAttr("operation", AttrValue::CreateFrom<AttrValue::STR>(operation_map[param.operation()]));
    }
    return SUCCESS;
}

// #### Obtains the processing function of the output tensor description. 
Status CaffeReductionInferShapeAndType(const ge::Operator& op, vector<ge::TensorDesc>& v_output_desc)
{
    auto tensorDesc      = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();
    int64_t axis = -1;
    
    ge::AttrValue axisAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("axis", axisAttrValue)) || (ge::GRAPH_SUCCESS != axisAttrValue.GetValue<AttrValue::INT>(axis)))
    {
        printf("Get axis failed!\n");
    }
    // In the OM model, all shape are supplemented to 4d. In this case, axis needs to be repaired to point to the original 2d.
    if (axis < 0) axis -= 2;

    if (axis < 0) axis += shape.GetDimNum();

    if (axis < 0 || axis >= shape.GetDimNum())
    {
        printf("invalid axis:%d, dim_size:%d\n", (int32_t)axis, (int32_t)shape.GetDimNum());
        return PARAM_INVALID;
    }
    int32_t dimsize = (int32_t)shape.GetDimNum();
    int32_t idx = 0;
    for(idx=axis; idx<dimsize; idx++)
    {
        shape.SetDim(idx, 1);
    }
    tensorDesc.SetShape(shape);
    v_output_desc.push_back(tensorDesc);

    return SUCCESS;

}


// build Te Binary file
Status CaffeReductionBuildTeBin(const ge::Operator& op, TEBinInfo& te_bin_info)
{
    std::string FilePath   = "";
    std::string FuncName   = "";
    std::string KernelName = "";
    std::string operation  = "";
    int64_t     axis       = -1;
    float       coeff      = 1;
    // ### Parses the operation parameter. 
    ge::AttrValue operationAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("operation", operationAttrValue)) || (ge::GRAPH_SUCCESS != operationAttrValue.GetValue<AttrValue::STR>(operation)))
    {
        // ### Add exception handling and maintenance information. 
        printf("GetOpAttr operation failed!\n");
    }

    // ### Parse the axis parameter. 
    ge::AttrValue axisAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("axis", axisAttrValue)) || (ge::GRAPH_SUCCESS != axisAttrValue.GetValue<AttrValue::INT>(axis)))
    {
        printf("GetOpAttr axis failed!\n");
    }
    // In the OM model, all shape are supplemented to 4d. In this case, axis needs to be repaired to point to the original 2d.
    if(axis < 0)
        axis -= 2;

    // ### Parse the coeff parameter. 
    ge::AttrValue coeffAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("coeff", coeffAttrValue)) || (ge::GRAPH_SUCCESS != coeffAttrValue.GetValue<AttrValue::FLOAT>(coeff)))
    {
        printf("GetOpAttr coeff failed!\n");
    }
    // ### Parse input tensor description 
    TensorDesc input_desc      = op.GetInputDesc(0);

    // ### Parse the input shape value and check whether the value is 4.
    if(input_desc.GetShape().GetDimNum() != 4)
    {
        printf("The shape size is %d, which is not 4!", (int32_t)input_desc.GetShape().GetDimNum());
        return FAILED;
    }
    FilePath   = "../operator/reduction";
    FuncName   = "reduction";
    KernelName = "Reduction";

    // i => int; s => string; f => dobule; O => bool, and bool value is Py_True or Py_False
    te::BuildTeCustomOp(te_bin_info.ddk_version, op.GetName(), FilePath, FuncName,
                    "(i,i,i,i), s, i, s, f, s", input_desc.GetShape().GetDim(0), input_desc.GetShape().GetDim(1),
                    input_desc.GetShape().GetDim(2), input_desc.GetShape().GetDim(3), "float16", axis, operation.c_str(), coeff,
                    KernelName.c_str());

    // set te op json to te_bin_info 
    te_bin_info.bin_file_path  = "./operator/kernel_meta/" + KernelName + ".o";
    te_bin_info.json_file_path = "./operator/kernel_meta/" + KernelName + ".json";

    return SUCCESS;
}

REGISTER_CUSTOM_OP("custom_reduction") //test_reduction is the type name of the operator in the OM model. It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("Reduction")  // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(CaffeReductionParseParams)  // Op parameters parse function
    .InferShapeAndTypeFn(CaffeReductionInferShapeAndType)       // Set output description and datatype function
    .TEBinBuildFn(CaffeReductionBuildTeBin)           // Build Te op binary function
    .ImplyType(ImplyType::TVM);        // Implementation type. Enumerated type, The options are as follows: TVM, AI_CPU.

}  // namespace domi
