/***************************************************************************************
*                      CopyRight (C) Hisilicon Co., Ltd.
*
*       Filename:  user_def_errorcode.h
*    Description:  User defined Error Code
*
*        Version:  1.0
*        Created:  2018-01-08 10:15:18
*         Author:
*
*       Revision:  initial draft;
**************************************************************************************/
#ifndef ERROR_CODE_H_
#define ERROR_CODE_H_

#include "hiaiengine/status.h"
#define MODID_CUSTOM   0x0301
enum
{
    HIAI_INVALID_INPUT_MSG_CODE=0x0301,
    HIAI_AI_MODEL_MANAGER_INIT_FAIL_CODE,
    HIAI_AI_MODEL_MANAGER_PROCESS_FAIL_CODE,
    HIAI_SEND_DATA_FAIL_CODE
};
HIAI_DEF_ERROR_CODE(MODID_CUSTOM, HIAI_ERROR, HIAI_INVALID_INPUT_MSG, \
    "invalid input message pointer");
HIAI_DEF_ERROR_CODE(MODID_CUSTOM, HIAI_ERROR, HIAI_AI_MODEL_MANAGER_INIT_FAIL, \
    "ai model manager init failed");
HIAI_DEF_ERROR_CODE(MODID_CUSTOM, HIAI_ERROR, HIAI_AI_MODEL_MANAGER_PROCESS_FAIL, \
    "ai model manager process failed");
HIAI_DEF_ERROR_CODE(MODID_CUSTOM, HIAI_ERROR, HIAI_SEND_DATA_FAIL, \
    "send data failed");
#endif //ERROR_CODE_H_

