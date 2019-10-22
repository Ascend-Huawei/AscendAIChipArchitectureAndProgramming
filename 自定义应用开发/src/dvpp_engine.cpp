/**
* @file dest_engines.cpp
*
* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "inc/dvpp_engine.h"
#include "inc/error_code.h"
#include <hiaiengine/log.h>
#include <hiaiengine/c_graph.h>
#include <hiaiengine/ai_memory.h>
#include <vector>
#include <dvpp/idvppapi.h>
#include <dvpp/Vpc.h>
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include "inc/sample_data.h"

int32_t DvppEngine::VpcProcess(const uint8_t* inputData,
                               uint32_t inputDataSize,
                               std::shared_ptr<std::string>& output_string_ptr)
{
    uint32_t inWidthStride = 1024;
    uint32_t inHeightStride = 684;
    uint32_t outWidthStride = 224;
    uint32_t outHeightStride = 224;
    uint32_t inBufferSize = inWidthStride * inHeightStride * 3 / 2; 
    uint32_t outBufferSize = outWidthStride * outHeightStride * 3 / 2; 

    uint8_t* inBuffer = const_cast<uint8_t*>(inputData);
    if (inBuffer == nullptr) {
        HIAI_ENGINE_LOG(this, HIAI_DVPP_MANAGER_PROCESS_FAIL, "inBuffer is null.");
        return HIAI_DVPP_MANAGER_PROCESS_FAIL;
    }

    uint8_t* outBuffer = reinterpret_cast<uint8_t*>(HIAI_DVPP_DMalloc(outBufferSize));
    if (outBuffer == nullptr) {
        HIAI_ENGINE_LOG(this, HIAI_DVPP_MANAGER_PROCESS_FAIL, "outBuffer is null.");
        return HIAI_DVPP_MANAGER_PROCESS_FAIL;
    }

    // Construct the input picture configuration.
    std::shared_ptr<VpcUserImageConfigure> imageConfigure(new VpcUserImageConfigure);
    imageConfigure->bareDataAddr = inBuffer;
    imageConfigure->bareDataBufferSize = inBufferSize;
    imageConfigure->widthStride = inWidthStride;
    imageConfigure->heightStride = inHeightStride;
    imageConfigure->inputFormat = INPUT_YUV420_SEMI_PLANNER_UV;
    imageConfigure->outputFormat = OUTPUT_YUV420SP_UV;
    imageConfigure->yuvSumEnable = false;
    imageConfigure->cmdListBufferAddr = nullptr;
    imageConfigure->cmdListBufferSize = 0;
    std::shared_ptr<VpcUserRoiConfigure> roiConfigure(new VpcUserRoiConfigure);
    roiConfigure->next = nullptr;
    VpcUserRoiInputConfigure* inputConfigure = &roiConfigure->inputConfigure;

    //Set the drawing area.
    inputConfigure->cropArea.leftOffset  = 0;
    inputConfigure->cropArea.rightOffset = 1023;
    inputConfigure->cropArea.upOffset    = 0;
    inputConfigure->cropArea.downOffset  = 683;
    VpcUserRoiOutputConfigure* outputConfigure = &roiConfigure->outputConfigure;
    outputConfigure->addr = outBuffer;
    outputConfigure->bufferSize = outBufferSize;
    outputConfigure->widthStride = outWidthStride;
    outputConfigure->heightStride = outHeightStride;

    // Set the map area.
    outputConfigure->outputArea.leftOffset  = 0;
    outputConfigure->outputArea.rightOffset = 223;
    outputConfigure->outputArea.upOffset    = 0;
    outputConfigure->outputArea.downOffset  = 223;

    imageConfigure->roiConfigure = roiConfigure.get();

    IDVPPAPI *pidvppapi = nullptr;
    int32_t ret = CreateDvppApi(pidvppapi);
    if (ret != 0) {
        HIAI_ENGINE_LOG(this, HIAI_DVPP_MANAGER_PROCESS_FAIL, "create dvpp api fail.");
        HIAI_DVPP_DFree(outBuffer);
        return HIAI_DVPP_MANAGER_PROCESS_FAIL;
    }
	dvppapi_ctl_msg dvppApiCtlMsg;
    dvppApiCtlMsg.in = reinterpret_cast<void*>(imageConfigure.get());
    dvppApiCtlMsg.in_size = sizeof(VpcUserImageConfigure);

    ret = DvppCtl(pidvppapi, DVPP_CTL_VPC_PROC, &dvppApiCtlMsg);
    if (ret != 0) {
        HIAI_ENGINE_LOG(this, HIAI_DVPP_MANAGER_PROCESS_FAIL, "call vpc dvppctl process faild!");
        ret = DestroyDvppApi(pidvppapi);
        HIAI_DVPP_DFree(outBuffer);
        return HIAI_DVPP_MANAGER_PROCESS_FAIL;
    } else {
        HIAI_ENGINE_LOG("call vpc dvppctl process success!");
        output_string_ptr = std::shared_ptr<std::string>(new std::string((char*)outBuffer, outBufferSize));
    }

    ret = DestroyDvppApi(pidvppapi);
    if (ret != 0) {
        HIAI_ENGINE_LOG(this, HIAI_DVPP_MANAGER_PROCESS_FAIL, "destroy dvpp api fail.");
        ret = HIAI_DVPP_MANAGER_PROCESS_FAIL;
    }

    return ret;
}

// Dvpp Engine
HIAI_IMPL_ENGINE_PROCESS("DvppEngine", DvppEngine, DVPP_ENGINE_INPUT_SIZE)
{
    // 获取host侧传过来的Vpc输入数据
    std::shared_ptr<EngineTransNewT> input_arg =
        std::static_pointer_cast<EngineTransNewT>(arg0);

    if (nullptr == input_arg) {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG, "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }

    // check input data size
    if (input_arg->buffer_size != 1024 * 684 * 3 / 2) {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG,
            "input message size (%u) not match, it should be 1050624", input_arg->buffer_size);
        return HIAI_INVALID_INPUT_MSG;
    }

    // call Vpc process
    std::shared_ptr<std::string> output_string_ptr = std::make_shared<std::string>();
    int32_t ret = VpcProcess(input_arg->trans_buff.get(), input_arg->buffer_size, output_string_ptr);
    if (ret != 0) {
        HIAI_ENGINE_LOG(this, HIAI_DVPP_MANAGER_PROCESS_FAIL, "call vpc process fail");
        char* outBuffer = const_cast<char*>(output_string_ptr->c_str());
        if (outBuffer != nullptr) {
            HIAI_DVPP_DFree(outBuffer);
        }
        return HIAI_DVPP_MANAGER_PROCESS_FAIL;
    }

    // 将Vpc结果输出到端口0
    hiai::Engine::SendData(0, "string", std::static_pointer_cast<void>(output_string_ptr));
    char* outBuffer = const_cast<char*>(output_string_ptr->c_str());
    if (outBuffer != nullptr) {
        HIAI_DVPP_DFree(outBuffer);
    }

    return HIAI_OK;
}
