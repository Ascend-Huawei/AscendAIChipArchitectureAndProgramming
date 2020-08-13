/**
* @file dest_engines.cpp
*
* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "inc/dest_engine.h"
#include "inc/error_code.h"
#include <hiaiengine/log.h>
#include <vector>
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>

// Dest Engine
HIAI_IMPL_ENGINE_PROCESS("DestEngine", DestEngine, DEST_ENGINE_INPUT_SIZE)
{
    // receive data
    std::shared_ptr<std::string> input_arg =
        std::static_pointer_cast<std::string>(arg0);

    if (nullptr == input_arg)
    {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG, "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }

    // send data to port 0
    hiai::Engine::SendData(0, "string", std::static_pointer_cast<void>(input_arg));

    return HIAI_OK;
}