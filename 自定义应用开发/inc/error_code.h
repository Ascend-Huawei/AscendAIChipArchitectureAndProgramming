/* 
* @file error_code.h
*
* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef ERROR_CODE_H_
#define ERROR_CODE_H_

#include "hiaiengine/status.h"
#define MODID_HIAIENGINE_SAMPLE 0x701
enum
{
	HIAI_INVALID_INPUT_MSG_CODE,
	HIAI_AI_MODEL_MANAGER_INIT_FAIL_CODE,
	HIAI_AI_MODEL_MANAGER_PROCESS_FAIL_CODE,
    HIAI_DVPP_MANAGER_PROCESS_FAIL_CODE,
	HIAI_SEND_DATA_FAIL_CODE
};
HIAI_DEF_ERROR_CODE(MODID_HIAIENGINE_SAMPLE, HIAI_ERROR, HIAI_INVALID_INPUT_MSG, \
    "invalid input message pointer");
HIAI_DEF_ERROR_CODE(MODID_HIAIENGINE_SAMPLE, HIAI_ERROR, HIAI_AI_MODEL_MANAGER_INIT_FAIL, \
    "ai model manager init failed");
HIAI_DEF_ERROR_CODE(MODID_HIAIENGINE_SAMPLE, HIAI_ERROR, HIAI_AI_MODEL_MANAGER_PROCESS_FAIL, \
    "ai model manager process failed");
HIAI_DEF_ERROR_CODE(MODID_HIAIENGINE_SAMPLE, HIAI_ERROR, HIAI_SEND_DATA_FAIL, \
    "send data failed");
HIAI_DEF_ERROR_CODE(MODID_HIAIENGINE_SAMPLE, HIAI_ERROR, HIAI_DVPP_MANAGER_PROCESS_FAIL, \
    "ai model manager init failed");
#endif //ERROR_CODE_H_
