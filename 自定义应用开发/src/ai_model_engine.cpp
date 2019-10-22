#include "inc/ai_model_engine.h"
#include "inc/error_code.h"
#include <hiaiengine/log.h>
#include <hiaiengine/ai_types.h>
#include "hiaiengine/ai_model_parser.h"
#include <vector>
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include <iostream>  

// Framework Engine
FrameworkerEngine::FrameworkerEngine():
    ai_model_manager_(nullptr)
{
}

HIAI_StatusT FrameworkerEngine::Init(const hiai::AIConfig& config,
   const  std::vector<hiai::AIModelDescription>& model_desc)
{
    hiai::AIStatus ret = hiai::SUCCESS;
    // init ai_model_manager_
    if (nullptr == ai_model_manager_)
    {
        ai_model_manager_ = std::make_shared<hiai::AIModelManager>();
    }
    std::cout<<"FrameworkerEngine Init"<<std::endl;

    for (int index = 0; index < config.items_size(); ++index)
    {

        const ::hiai::AIConfigItem& item = config.items(index);
        // loading model
        if(item.name() == "model_path")
        {
            const char* model_path = item.value().data();
            std::vector<hiai::AIModelDescription> model_desc_vec;
            hiai::AIModelDescription model_desc_;
            model_desc_.set_path(model_path);
            model_desc_.set_key("");
            model_desc_vec.push_back(model_desc_);
            ret = ai_model_manager_->Init(config, model_desc_vec);

            if (hiai::SUCCESS != ret)
            {
                HIAI_ENGINE_LOG(this, HIAI_AI_MODEL_MANAGER_INIT_FAIL, "[DEBUG] fail to init ai_model");
                return HIAI_AI_MODEL_MANAGER_INIT_FAIL;
            }
        }
    }

    return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("FrameworkerEngine", FrameworkerEngine, \
    FRAMEWORK_ENGINE_INPUT_SIZE)
{
    hiai::AIStatus ret = hiai::SUCCESS;
    HIAI_StatusT hiai_ret = HIAI_OK;
    // receive data
    std::shared_ptr<std::string> input_arg =
        std::static_pointer_cast<std::string>(arg0);
    if (nullptr == input_arg)
    {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG, "[DEBUG] input arg is invalid");
        return HIAI_INVALID_INPUT_MSG;
    }
    std::cout<<"FrameworkerEngine Process"<<std::endl;

    //  prapare for calling the process of ai_model_manager_
    std::vector<std::shared_ptr<hiai::IAITensor>> input_data_vec;

    uint32_t len = 75264;


    std::cout << "HIAIAippOp::Go to process" << std::endl;
    std::shared_ptr<hiai::AINeuralNetworkBuffer> neural_buffer = std::shared_ptr<hiai::AINeuralNetworkBuffer>(new hiai::AINeuralNetworkBuffer());//std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(input_data);
    neural_buffer->SetBuffer((void*)(input_arg->c_str()), (uint32_t)(len));
    std::shared_ptr<hiai::IAITensor> input_data = std::static_pointer_cast<hiai::IAITensor>(neural_buffer);
    input_data_vec.push_back(input_data);

    //  call Process and inference
    hiai::AIContext ai_context;
    std::vector<std::shared_ptr<hiai::IAITensor>> output_data_vec;
    ret = ai_model_manager_->CreateOutputTensor(input_data_vec, output_data_vec);
    if (hiai::SUCCESS != ret)
    {
        HIAI_ENGINE_LOG(this, HIAI_AI_MODEL_MANAGER_PROCESS_FAIL, "[DEBUG] fail to process ai_model");
        return HIAI_AI_MODEL_MANAGER_PROCESS_FAIL;
    }

    ret = ai_model_manager_->Process(ai_context, input_data_vec, output_data_vec, 0);

    if (hiai::SUCCESS != ret)
    {

        HIAI_ENGINE_LOG(this, HIAI_AI_MODEL_MANAGER_PROCESS_FAIL, "[DEBUG] fail to process ai_model");
        return HIAI_AI_MODEL_MANAGER_PROCESS_FAIL;
    }
    std::cout<<"[DEBUG] output_data_vec size is "<< output_data_vec.size()<<std::endl;
    for (uint32_t index = 0; index < output_data_vec.size(); index++)
    {
        // send data of inference to destEngine
        std::shared_ptr<hiai::AINeuralNetworkBuffer> output_data = std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(output_data_vec[index]);
        std::shared_ptr<std::string> output_string_ptr = std::shared_ptr<std::string>(new std::string((char*)output_data->GetBuffer(), output_data->GetSize()));

        hiai_ret = SendData(0, "string", std::static_pointer_cast<void>(output_string_ptr));
        if (HIAI_OK != hiai_ret)
        {
            HIAI_ENGINE_LOG(this, HIAI_SEND_DATA_FAIL, "fail to send data");
        }
    }
    return HIAI_OK;
}
