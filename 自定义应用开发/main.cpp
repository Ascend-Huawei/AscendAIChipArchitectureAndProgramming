/* 
* @file main.cpp
*
* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include "hiaiengine/api.h"
#include "inc/error_code.h"
#include "inc/tensor.h"
#include "inc/sample_data.h"
#include "hiaiengine/ai_memory.h"

static const std::string test_src_file = "./test_data/data/dog_1024x684.yuv420sp";  // test data file
static const std::string test_dest_filename =
    "./test_data/matrix_dvpp_framework_test_result";  // output result file
static const std::string graph_config_proto_file =
    "./test_data/config/sample.prototxt";  // Graph config file
static const std::string GRAPH_MODEL_PATH =
        "./test_data/model/resnet18.om";
static const uint32_t GRAPH_ID = 100;
static const uint32_t SRC_ENGINE_ID = 1000;
static const uint32_t SRC_PORT_ID = 0;
static const uint32_t DEST_ENGINE_ID = 1002;
static const uint32_t DEST_PORT_ID = 0;
static std::mutex local_test_mutex;
static std::condition_variable local_test_cv_;
static bool is_test_result_ready = false;
static const int MAX_SLEEP_TIMER = 30 * 60;

// Defines Output shape
const std::vector<uint32_t> DATA_NUM = {1000};
const int MAX_SLEEP_TIMES = 16;

// Define Data Recv Interface
class DdkDataRecvInterface : public hiai::DataRecvInterface
{
public:
    DdkDataRecvInterface(const std::string& filename) :
        file_name_(filename)
    {
    }

    /**
    * @ingroup hiaiengine
    * @brief receive data and save
    * @param [in]input message
    * @return HIAI Status
    */
    HIAI_StatusT RecvData(const std::shared_ptr<void>& message)
    {
        // receive data
        std::shared_ptr<std::string> data =
            std::static_pointer_cast<std::string>(message);
        if (nullptr == data)
        {
            HIAI_ENGINE_LOG("Fail to receive data");
            return HIAI_INVALID_INPUT_MSG;
        }

        ddk::Tensor<float> num;
        num.fromarray(reinterpret_cast<float*>(const_cast<char*>(data->c_str())), DATA_NUM);
        (void)num.dump(file_name_);

        return HIAI_OK;
    }
private:

    std::string file_name_;
};

// function for read bin file
char* ReadBinFile(const char *file_name, uint32_t *fileSize)
{
    std::filebuf *pbuf;
    std::ifstream filestr; 
    size_t size;
    char * buffer;
    filestr.open(file_name, std::ios::binary);
    if (!filestr) 
    {
        return NULL;
    } 

    pbuf = filestr.rdbuf();
    size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
    pbuf->pubseekpos(0, std::ios::in);
    HIAI_StatusT getRet = hiai::HIAIMemory::HIAI_DMalloc(size, (void*&)buffer, 10000);
    if (HIAI_OK != getRet || nullptr == buffer)
    {
        buffer = new char[size];
    }

    if (NULL == buffer)
    {
        filestr.close();
        return NULL;
    }

    pbuf->sgetn(buffer, size);
    *fileSize = size;

    filestr.close();
    return buffer;
}

// Init and create graph
HIAI_StatusT HIAI_InitAndStartGraph()
{
    // Step1: HiAi init
    HIAI_StatusT status = HIAI_Init(0);

    // Step2: create Graph
    status = hiai::Graph::CreateGraph(graph_config_proto_file);
    if (status != HIAI_OK)
    {
        HIAI_ENGINE_LOG(status, "Fail to start graph");
        return status;
    }

    // Step3: obtain graph instance
    std::shared_ptr<hiai::Graph> graph = hiai::Graph::GetInstance(GRAPH_ID);
    if (nullptr == graph)
    {
        HIAI_ENGINE_LOG("Fail to get the graph-%u", GRAPH_ID);
        return status;
    }

    hiai::EnginePortID target_port_config;
    target_port_config.graph_id = GRAPH_ID;
    target_port_config.engine_id = DEST_ENGINE_ID;
    target_port_config.port_id = DEST_PORT_ID;

    // set functor for receive data
    graph->SetDataRecvFunctor(target_port_config,
        std::shared_ptr<DdkDataRecvInterface>(
            new DdkDataRecvInterface(test_dest_filename)));

    return HIAI_OK;
}

// check if target file exist or not
static bool file_exist(const std::string& file_name)
{
    std::ifstream f(file_name.c_str());
    return f.good();
}

void deleteNothing(void* ptr)
{
    // do nothing
}

// oversee result file is already generate or not
void checkDestFileExist()
{
    for (int i = 0; i < MAX_SLEEP_TIMER; ++i) {
        if (file_exist(test_dest_filename)) {
            std::unique_lock <std::mutex> lck(local_test_mutex);
            is_test_result_ready = true;
            printf("File %s generated\n", test_dest_filename.c_str());
            HIAI_ENGINE_LOG("Check Result success");
            return;
        }
        printf("Check Result, go into sleep 1 sec\n");
        HIAI_ENGINE_LOG("Check Result, go into sleep 1 sec");
        sleep(1);
    }
    printf("Check Result failed, timeout\n");
}

// main
int main(int argc, char* argv[])
{
    printf("========== Test Start ==========\n");
    HIAI_StatusT ret = HIAI_OK;
    // Perform Initialziation 
    remove(test_dest_filename.c_str());

    for (int i = 0; i < MAX_SLEEP_TIMES; ++i) {
        if (file_exist(GRAPH_MODEL_PATH)) {
            printf("File %s is ready\n", GRAPH_MODEL_PATH.c_str());
            break;
        }
        sleep(1);
        if (i == MAX_SLEEP_TIMES-1) {
            printf("model file:%s is not existence, timeout\n", GRAPH_MODEL_PATH.c_str());
        }
    }

    // call functor for creating graph
    ret = HIAI_InitAndStartGraph();
    if (HIAI_OK != ret)
    {
        HIAI_ENGINE_LOG("Fail to start graph");;
        return -1;
    }
    printf("Init and start graph succeed\n");
    // obtain graph instance
    std::shared_ptr<hiai::Graph> graph = hiai::Graph::GetInstance(GRAPH_ID);
    if (nullptr == graph)
    {
        HIAI_ENGINE_LOG("Fail to get the graph-%u", GRAPH_ID);
        return -1;
    }

    // send data to SrcEngine in port 0
    hiai::EnginePortID engine_id;
    engine_id.graph_id = GRAPH_ID;
    engine_id.engine_id = SRC_ENGINE_ID;
    engine_id.port_id = SRC_PORT_ID;
    uint32_t size = 0;
    char* dataBuffer = ReadBinFile(test_src_file.c_str(), &size);

    std::shared_ptr<EngineTransNewT> tmp_raw_data_ptr = std::make_shared<EngineTransNewT>();
    tmp_raw_data_ptr->buffer_size = size;
    tmp_raw_data_ptr->trans_buff.reset((unsigned char*)dataBuffer, deleteNothing);

    graph->SendData(engine_id, "EngineTransNewT",
            std::static_pointer_cast<void>(tmp_raw_data_ptr));
    // wait for the result file
    std::thread check_thread(checkDestFileExist);
    check_thread.join();

    if (is_test_result_ready) {
        printf("========== Test Succeed ==========\n");
    } else {
        printf("========== Test Failed ==========\n");
    }
    // end the graph and destory graph
    hiai::Graph::DestroyGraph(GRAPH_ID);
    return 0;
}
