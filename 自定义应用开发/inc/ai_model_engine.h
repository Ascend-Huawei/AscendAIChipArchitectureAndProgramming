
#ifndef FRAMEWORKER_ENGINE_H_
#define FRAMEWORKER_ENGINE_H_
//#include "hiaiengine/data_type_reg.h"
//#include "hiaiengine/data_type.h"
#include "hiaiengine/api.h"
#include "hiaiengine/ai_model_manager.h"

#define FRAMEWORK_ENGINE_INPUT_SIZE  1
#define FRAMEWORK_ENGINE_OUTPUT_SIZE  1
using hiai::Engine;

// Framework Engine
class FrameworkerEngine : public Engine {
public:
    HIAI_StatusT Init(const hiai::AIConfig& config,
       const std::vector<hiai::AIModelDescription>& model_desc);

    FrameworkerEngine();
    /**
    * @ingroup hiaiengine
    * @brief HIAI_DEFINE_PROCESS : 重载Engine Process处理逻辑
    * @[in]: 定义一个输入端口，一个输出端口
    */
    HIAI_DEFINE_PROCESS(FRAMEWORK_ENGINE_INPUT_SIZE, FRAMEWORK_ENGINE_OUTPUT_SIZE)
private:
    std::shared_ptr<hiai::AIModelManager> ai_model_manager_;
};
#endif //FRAMEWORKER_ENGINE_H_