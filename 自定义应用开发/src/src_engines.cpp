/**
* @file src_engines.cpp
*
* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <hiaiengine/log.h>
#include <vector>
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include "inc/src_engine.h"
#include "inc/error_code.h"
#include "inc/sample_data.h"

// Source Engine
HIAI_IMPL_ENGINE_PROCESS("SrcEngine", SrcEngine, SOURCE_ENGINE_INPUT_SIZE)
{
    // receive data
    if (nullptr == arg0)
    {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG, "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }

    // send tata to port 0
    hiai::Engine::SendData(0, "EngineTransNewT", arg0);

    return HIAI_OK;
}