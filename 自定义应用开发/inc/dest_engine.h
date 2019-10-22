/* 
* @file dest_engine.h
*
* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef DEST_ENGINE_H_
#define DEST_ENGINE_H_

#include "hiaiengine/api.h"

#define DEST_ENGINE_INPUT_SIZE  1
#define DEST_ENGINE_OUTPUT_SIZE  1

using hiai::Engine;

// Dst Engine
class DestEngine : public Engine {
    /**
    * @ingroup hiaiengine
    * @brief HIAI_DEFINE_PROCESS : 重载Engine Process处理逻辑
    * @[in]: 定义一个输入端口，一个输出端口
    */
    HIAI_DEFINE_PROCESS(DEST_ENGINE_INPUT_SIZE, DEST_ENGINE_OUTPUT_SIZE)
};
#endif //DEST_ENGINE_H_