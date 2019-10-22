
#ifndef DVPP_ENGINE_H_
#define DVPP_ENGINE_H_
#include "hiaiengine/api.h"
#include "hiaiengine/ai_model_manager.h"

#define DVPP_ENGINE_INPUT_SIZE  1
#define DVPP_ENGINE_OUTPUT_SIZE  1
using hiai::Engine;

// Dvpp Engine
class DvppEngine : public Engine {
public:
    /**
    * @ingroup hiaiengine
    * @brief HIAI_DEFINE_PROCESS : 重载Engine Process处理逻辑
    * @[in]: 定义一个输入端口，一个输出端口
    */
    HIAI_DEFINE_PROCESS(DVPP_ENGINE_INPUT_SIZE, DVPP_ENGINE_OUTPUT_SIZE)
private:
    int32_t VpcProcess(const uint8_t* inputData, uint32_t inputDataSize, std::shared_ptr<std::string>& output_string_ptr);
};
#endif //DVPP_ENGINE_H_