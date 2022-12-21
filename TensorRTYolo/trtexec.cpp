#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvOnnxParser.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include <pthread.h>
#include <thread>
#include<mutex>
#include</usr/src/tensorrt/samples/trtexec/utils.h>
#include <cuda_profiler_api.h>
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace nvonnxparser;
using namespace cv;
using namespace std;
using namespace sample;
std::mutex mu;
const std::string gSampleName = "TensorRT.trtexec";
//ICudaEngine& engine;
struct Params
{
    std::string deployFile{};
    std::string modelFile{};
    std::string engine{};
    std::string saveEngine{};
    std::string loadEngine{};
    std::string calibrationCache{"CalibrationTable"};
    std::string uffFile{};
    std::string onnxModelFile{};
    std::vector<std::string> inputs{};
    std::vector<std::string> outputs{};
    std::vector<std::pair<std::string, Dims3>> uffInputs{};
    int device{0};
    int batchSize{1};
    int workspaceSize{16};
    int iterations{10};
    int avgRuns{10};
    int useDLACore{-1};
    bool safeMode{false};
    bool fp16{false};
    bool int8{false};
    bool verbose{false};
    bool allowGPUFallback{true};
    float pct{99};
    bool useSpinWait{false};
    bool dumpOutput{false};
    bool help{false};
} gParams;

inline int volume(Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

std::map<std::string, Dims3> gInputDimensions;

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}

float percentile(float percentage, std::vector<float>& times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        std::sort(times.begin(), times.end());
        return times[all == exclude ? 0 : all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}
class InferenceEngine
{
public:
        InferenceEngine(const string& model_file,
                const string& trained_file);

        ICudaEngine* deSerializeEngine(const std::string& engineFile);

        ~InferenceEngine();

        ICudaEngine* Get() const
        {
                return engine_;
        }

private:
        ICudaEngine* engine_;
};

class RndInt8Calibrator : public IInt8EntropyCalibrator2
{
public:
    RndInt8Calibrator(int totalSamples, std::string cacheFile)
        : mTotalSamples(totalSamples)
        , mCurrentSample(0)
        , mCacheFile(cacheFile)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        for (auto& elem : gInputDimensions)
        {
            int elemCount = volume(elem.second);

            std::vector<float> rnd_data(elemCount);
            for (auto& val : rnd_data)
                val = distribution(generator);

            void* data;
            CHECK(cudaMalloc(&data, elemCount * sizeof(float)));
            CHECK(cudaMemcpy(data, &rnd_data[0], elemCount * sizeof(float), cudaMemcpyHostToDevice));

            mInputDeviceBuffers.insert(std::make_pair(elem.first, data));
        }
    }

    ~RndInt8Calibrator()
    {
        for (auto& elem : mInputDeviceBuffers)
            CHECK(cudaFree(elem.second));
    }

    int getBatchSize() const override
    {
        return 1;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (mCurrentSample >= mTotalSamples)
            return false;

        for (int i = 0; i < nbBindings; ++i)
            bindings[i] = mInputDeviceBuffers[names[i]];

        ++mCurrentSample;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCacheFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void*, size_t) override
    {
    }

private:
    int mTotalSamples;
    int mCurrentSample;
    std::string mCacheFile;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
};

void configureBuilder(IBuilder* builder, RndInt8Calibrator& calibrator)
{
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(static_cast<size_t>(gParams.workspaceSize) << 20);
    builder->setFp16Mode(gParams.fp16);

    if (gParams.int8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(&calibrator);
    }

    if (gParams.safeMode)
    {
        builder->setEngineCapability(gParams.useDLACore >= 0 ? EngineCapability::kSAFE_DLA : EngineCapability::kSAFE_GPU);
    }
}
/*
ICudaEngine* caffeToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    if (builder == nullptr)
    {
        return nullptr;
    }

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
                                                              gParams.modelFile.empty() ? 0 : gParams.modelFile.c_str(),
                                                              *network,
                                                              DataType::kFLOAT);

    if (!blobNameToTensor)
    {
        return nullptr;
    }

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gParams.inputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
        gLogInfo << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (blobNameToTensor->find(s.c_str()) == nullptr)
        {
            gLogError << "could not find output blob " << s << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getOutput(i)->getDimensions());
        gLogInfo << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
                 << dims.d[2] << std::endl;
    }

    // Build the engine
    RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    configureBuilder(builder, calibrator);

    samplesCommon::enableDLA(builder, gParams.useDLACore, gParams.allowGPUFallback);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
    {
        gLogError << "could not build engine" << std::endl;
    }

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}
*/
ICudaEngine* uffToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(sample::gLogger.getTRTLogger());
    if (builder == nullptr)
    {
        return nullptr;
    }

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    IUffParser* parser = createUffParser();

    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (!parser->registerOutput(s.c_str()))
        {
            gLogError << "Failed to register output " << s << std::endl;
            return nullptr;
        }
    }

    // specify which tensors are inputs (and their dimensions)
    for (auto& s : gParams.uffInputs)
    {
        if (!parser->registerInput(s.first.c_str(), s.second, UffInputOrder::kNCHW))
        {
            gLogError << "Failed to register input " << s.first << std::endl;
            return nullptr;
        }
    }

    if (!parser->parse(gParams.uffFile.c_str(), *network))
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gParams.inputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }

    // Build the engine
    RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    configureBuilder(builder, calibrator);

    //samplesCommon::enableDLA(builder, gParams.useDLACore);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        gLogError << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

ICudaEngine* onnxToTRTModel()
{
   cout<<"We have entered this function"<<endl;
    // create the builder
    IBuilder* builder = createInferBuilder(sample::gLogger.getTRTLogger());
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    if (builder == nullptr)
    {
        return nullptr;
    }
   // nvinfer1::INetworkDefinition* network = builder->createNetwork();
    

    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig(); 
//    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse the onnx model to populate the network, then set the outputs
    IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
    if ( !parser->parseFromFile( gParams.onnxModelFile.c_str(), static_cast<int>( sample::gLogger.getReportableSeverity() ) ) )
    {
        gLogError << "failed to parse onnx file" << std::endl;
        return nullptr;
    }
    cout<<"The number of inputs are "<<network->getNbInputs()<<endl;
    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }
    cout<<"Yes dude"<<endl;

    // Build the engine
    RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    configureBuilder(builder, calibrator);
     //if (m_DeviceType == "kDLA") {
     //  builder->setDefaultDeviceType(nvinfer1::DeviceType::kDLA, gParams.allowGPUFallback);
    //}    
     cout<<"Here we go"<<endl;
     samplesCommon::enableDLA(builder, config, gParams.useDLACore, gParams.allowGPUFallback);

//     samplesCommon::enableDLA(builder, 0 ,gParams.allowGPUFallback);
     
	cout<<"Damn could not"<<endl;

     //samplesCommon::enableDLA(builder, gParams.DLACore);
     //samplesCommon::enableDLA(builder->get(), config->get(), gParams->dlaCore);
	ICudaEngine* engine= builder->buildEngineWithConfig(*network,*config);
//    ICudaEngine* engine = builder->buildCudaEngine(*network);
   cout<<"The engine has been built I guess"<<endl;
    if (engine == nullptr)
    {
        gLogError << "could not build engine" << std::endl;
    }

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

void doInference(ICudaEngine& engine)
{ 

   cout<<"This is going good:"<<endl;






//static const char* kINPUT_BLOB_NAME = "data";            // Input blob name
//static const char* kOUTPUT_BLOB_NAME0 = "detection_out"; // Output blob name
//static const char* kOUTPUT_BLOB_NAME1 = "keep_count";
  //  IExecutionContext* context = engine.createExecutionContext();
    // int input_index = engine.getBindingIndex(kINPUT_BLOB_NAME);
     //cout<<"The input index is"<<input_index<<endl;
     //DimsCHW dims = static_cast<DimsCHW&&>(engine.getBindingDimensions(input_index));
     //size_t size = 1 * dims.c() * dims.h() * dims.w() * sizeof(float);
     //std::cout << "size of buff = " << size << std::endl;
    // Use an aliasing shared_ptr since we don't want engine to be deleted when bufferManager goes out of scope.
    //std::shared_ptr<ICudaEngine> emptyPtr{};
    //std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &engine);
    //samplesCommon::BufferManager bufferManager(aliasPtr, gParams.batchSize);
    //std::vector<void*> buffers = bufferManager.getDeviceBindings();
   //  cout<<"Hey the Bufffers are"<<buffers[0]<<endl;

static const int INPUT_C = 3;
static const int INPUT_H = 416;
static const int INPUT_W = 416;

void* buffers[3];


IExecutionContext *context = engine.createExecutionContext();

  
  int inputBindingIndex, outputBindingIndex,outputBindingIndex1;
  inputBindingIndex = engine.getBindingIndex("000_net");
  outputBindingIndex = engine.getBindingIndex("016_convolutional");
  outputBindingIndex1 = engine.getBindingIndex("023_convolutional");



cout<<"The input binding index is"<<inputBindingIndex<<endl;
cout<<"The output binding index is "<<outputBindingIndex<<endl;




  if (inputBindingIndex < 0)
  {
    cout << "Invalid input name." << endl;
    return;
  }

  if (outputBindingIndex < 0)
  {
    cout << "Invalid output name." << endl;
    return;
  }


CHECK(cudaMalloc(&buffers[inputBindingIndex], gParams.batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));



/*
   Dims inputDims, outputDims;
  inputDims = engine.getBindingDimensions(inputBindingIndex);
  outputDims = engine.getBindingDimensions(outputBindingIndex);
  int inputWidth, inputHeight;
  inputHeight = inputDims.d[1];
  inputWidth = inputDims.d[2];

*/
    cout << "Preprocessing input..." << endl;
  cv::Mat image = cv::imread("/usr/src/tensorrt/samples/trtexec/sample.jpg", cv::IMREAD_COLOR);

  if (image.data == NULL)
  {
    cout << "Could not read image from file." << endl;
    return;
  }

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
  cv::resize(image, image, cv::Size(INPUT_W, INPUT_H));


Dims outputDims=engine.getBindingDimensions(outputBindingIndex);
Dims outputDims1=engine.getBindingDimensions(outputBindingIndex1);

float *inputDataHost, *outputDataHost;
size_t  numOutput,numOutput1;
  //numInput = numTensorElements(inputDims);
  numOutput = numTensorElements(outputDims);
  numOutput1= numTensorElements(outputDims1);
cout<<"The size here is" <<numOutput<<"The other size is"<<numOutput1<<endl;
cudaError_t st = cudaMalloc(&buffers[outputBindingIndex], numOutput * sizeof(float));
 st = cudaMalloc(&buffers[outputBindingIndex1], numOutput1 * sizeof(float));

if(st!=cudaSuccess){
   cout<<"Memory could not be allocated"<<endl;
}
 //outputDataHost = (float*) malloc(numOutput * sizeof(float));
float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f };

//float* data = new float[1*size[0]*size[1]*size[2]];
cout<<"Here also"<<endl;
    /*for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                 data[c*width*height + h*w] =  ptr[ width*channel*h + w*channel + c ] - pixelMean[c];
             //           it++;

               // data[ c*width*height + h*width + w ] = ptr[ width*channel*h + w*channel + c ];
            }
        }
    }
cout<<"After this"<<data<<endl;*/
//cout<<"The input dimensions are "<<inputDims<<endl;
//cout<<"The output dimensions are"<<outputDims<<endl;
/*
float *inputDataHost, *outputDataHost;
  size_t numInput, numOutput;
  numInput = numTensorElements(inputDims);
  numOutput = numTensorElements(outputDims);

cout<<"The no of inputs are"<<numInput<<endl;
cout<<"The no of outputs are"<<numOutput<<endl;

  inputDataHost = (float*) malloc(numInput * sizeof(float));
  outputDataHost = (float*) malloc(numOutput * sizeof(float));
*/

//void* buffers[2];


float* data = new float[gParams.batchSize*INPUT_C*INPUT_H*INPUT_W];
cv::Mat_<cv::Vec3f>::iterator it;
	unsigned volChl = 416*416;
	for (int c = 0; c < 3; ++c)                              
	{
		cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();	//cv::Vec3f not working - reason still unknown...
		// the color image to input should be in BGR order
		for (unsigned j = 0; j < INPUT_H*INPUT_W; ++j)
		{
                        
                        

                        //cout<<"Data is"<<data[j]<<endl;
			//OpenCV read in frame as BGR format, by default, thus need only deduct the mean value
			data[c*INPUT_W*INPUT_H + j] = float((*it)[c]) - pixelMean[c];
			it++;
		}

	}

cout<<"The data is "<<data<<endl;
cudaStream_t stream;
CHECK(cudaStreamCreate(&stream));
cout<<"After stream"<<endl;


//cudaStreamS
/*cudaError_t cudaStatus=cudaMemcpyAsync(buffers[inputBindingIndex], data, 1 * INPUT_W * INPUT_H * INPUT_C * sizeof(float),
 cudaMemcpyHostToDevice,stream);
if(cudaStatus!=cudaSuccess){
cout<<"You will be fired"<<endl;
}
*/
cout<<"After copy"<<endl;













//cvImageToTensor(image, inputDataHost, inputDims);
//cout<<"The initial input is "<<inputDataHost<<endl;
//preprocessVgg(inputDataHost, inputDims);
//cout<<"The inputdatahost is "<<inputDataHost<<endl;




//void* buffers[2];
/*

CHECK(cudaMalloc(&buffers[inputBindingIndex], numInput * sizeof(float)));
CHECK(cudaMalloc(&buffers[outputBindingIndex], numOutput * sizeof(float))); 



 cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

//cudaStream_t stream;
//CHECK(cudaStreamCreate(&stream));

float *outputDataDevice;
cudaMalloc((void**)&outputDataDevice, numOutput * sizeof(float));
buffers[1]=(void*)outputDataDevice;
cout<<"The outputdata is"<<outputDataHost<<endl;



*/


//CHECK(cudaMemcpyAsync(buffers[inputBindingIndex], data, numInput * sizeof(float), cudaMemcpyHostToDevice, stream));

//void *bindings[2]
//float *outputDataDevice;
//cudaMalloc((void**)&outputDataDevice, numOutput * sizeof(float));
//buffers[1]=(void*)outputDataDevice;
//cout<<"The outputdata is"<<outputDataHost<<endl;

cudaStreamSynchronize(stream);
//bindings[0]=(void*)inputDataDevice;

///cout<<"The inputdatadevice is"<<inputDataDevice<<endl;
//out<<"The binding is"<<bindings[0]<<endl;
//bool var=context->execute(1,&bindings[0]);
constexpr int ITERATIONS = 100;
cudaEvent_t start;
cudaEvent_t end;
 double totalTime = 0.0;
cudaEventCreate(&start);
cudaEventCreate(&end);



cout << "execute\n";
cudaStreamSynchronize(stream);
size_t output_size = numOutput;//output_dim_.c() * output_dim_.h() * output_dim_.w();
std::vector<float> output(output_size);


size_t output_size1 = numOutput1;//output_dim_.c() * output_dim_.h() * output_dim_.w();
std::vector<float> output1(output_size1);
cout<<"Before loop"<<endl;
for(int i=0;i<ITERATIONS;i++){
float elapsedTime;
cout<<"Before event start"<<endl;
cudaEventCreate(&start);
cudaEventCreate(&end);
cudaError_t st = cudaEventRecord(start,stream);
if(st!=cudaSuccess){
cout<<"There is an error"<<endl;
}
cout<<"Yes dude"<<endl;
// auto t_start = std::chrono::high_resolution_clock::now();
cudaError_t cudaStatus=cudaMemcpyAsync(buffers[inputBindingIndex], data, gParams.batchSize * INPUT_W * INPUT_H * INPUT_C * sizeof(float),
 cudaMemcpyHostToDevice,stream);
bool var=context->enqueue(gParams.batchSize, buffers, stream, nullptr);
 cudaStatus=cudaMemcpyAsync(output.data(), buffers[outputBindingIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost,
stream);
cout<<"Reached before outout"<<endl;
cudaStatus=cudaMemcpyAsync(output1.data(), buffers[outputBindingIndex1], output_size1 * sizeof(float),
 cudaMemcpyDeviceToHost,stream);

cudaStreamSynchronize(stream);
cudaEventRecord(end, stream);
cudaEventSynchronize(end);
cudaStreamSynchronize(stream);
cudaEventElapsedTime(&elapsedTime, start, end);
totalTime += elapsedTime;
cout << "Inference batch size " << gParams.batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << endl;
//if(var==){
  //cout<<"There is something wrong going on"<<endl;

//}
//float *outputDataDevice;
//cudaMalloc((void**)&outputDataDevice, numOutput * sizeof(float));
//buffers[1]=(void*)outputDataDevice;
//cout<<"The outputdata is"<<outputDataHost<<endl;

cudaStreamSynchronize(stream);
/*size_t output_size = numOutput;//output_dim_.c() * output_dim_.h() * output_dim_.w();
std::vector<float> output(output_size);

 cudaStatus=cudaMemcpyAsync(output.data(), buffers[outputBindingIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost,
stream);
if(cudaStatus!=cudaSuccess){
 cout<<"Hell you dint transfer the data back"<<endl;

}
if (cudaStatus != cudaSuccess)
       throw std::runtime_error("could not copy output layer back to host");

size_t output_size1 = numOutput1;//output_dim_.c() * output_dim_.h() * output_dim_.w();
std::vector<float> output1(output_size1);

cudaStatus=cudaMemcpyAsync(output1.data(), buffers[outputBindingIndex1], output_size1 * sizeof(float),
 cudaMemcpyDeviceToHost,stream);
cout<<"The output is"<<outputDataHost<<endl;

cudaStreamSynchronize(stream);
cout<<"We entered this fucking area"<<endl;

/*
cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputBindingIndex]));
	CHECK(cudaFree(buffers[outputBindingIndex]));*/
// float *inputDataDevice, *outputDataDevice;

//cudaError_t cudaStatus;

// Choose which GPU to run on, change this on a multi-GPU system.
/*
cudaStatus = cudaSetDevice(0);
if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "CUDA failed!");
    return;
}


 cudaStatus= cudaMalloc((void**)&inputDataDevice, numInput * sizeof(float));
 if (cudaStatus == cudaSuccess) {
    cout<<"This is going wrog somehwere"<<endl;
    fprintf(stderr, "cudaMalloc failed!");
}
  cudaMalloc((void**)&outputDataDevice, numOutput * sizeof(float));
  cudaMemcpy(inputDataDevice, inputDataHost, numInput * sizeof(float), cudaMemcpyHostToDevice);
  void *bindings[2];
  bindings[inputBindingIndex] = (void*) inputDataDevice;
  bindings[outputBindingIndex] = (void*) outputDataDevice;
 cout<<"The bindings are"<<bindings[inputBindingIndex]<<endl;

*/
  /* execute engine */
  //cout << "Executing inference engine..." << endl;
  //const int kBatchSize = 1;
//  context->execute(kBatchSize,&bindings[inputBindingIndex]);

 //std::shared_ptr<ICudaEngine> emptyPtr{};
   // std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &engine);

//samplesCommon::BufferManager bufferManager(aliasPtr, gParams.batchSize);










   // float *inputDataDevice, *outputDataDevice;
   // void *inputDataDevice=malloc(numInput * sizeof(void*));
//std::vector<void*> buffers = bufferManager.getDeviceBindings();

//void* buffers = malloc(engine->getNbBindings() * sizeof(void*));









  //   cudaStream_t stream;
   // CHECK(cudaStreamCreate(&stream));
    /*cudaEvent_t start, end;
    unsigned int cudaEventFlags = gParams.useSpinWait ? cudaEventDefault : cudaEventBlockingSync;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventFlags));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventFlags));

    std::vector<float> times(gParams.avgRuns);
    for (int j = 0; j < gParams.iterations; j++)
    {
        float totalGpu{0}, totalHost{0}; // GPU and Host timers
        for (int i = 0; i < gParams.avgRuns; i++)
        {
            cout<<"Yeah I am in the loop"<<endl;
            auto tStart = std::chrono::high_resolution_clock::now();
            cudaEventRecord(start, stream);
            bool var=context->enqueue(1, &buffers[0], stream, nullptr);

            //context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);

            auto tEnd = std::chrono::high_resolution_clock::now();
            totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[i] = ms;
            totalGpu += ms;
        }
        totalGpu /= gParams.avgRuns;
        totalHost /= gParams.avgRuns;
        gLogInfo << "Average over " << gParams.avgRuns << " runs is " << totalGpu << " ms (host walltime is " << totalHost
                 << " ms, " << static_cast<int>(gParams.pct) << "\% percentile time is " << percentile(gParams.pct, times) << ")." << std::endl;
    }

/*    if (gParams.dumpOutput)
    {
        bufferManager.copyOutputToHost();
        int nbBindings = engine.getNbBindings();
        for (int i = 0; i < nbBindings; i++)
        {
            if (!engine.bindingIsInput(i))
            {
                const char* tensorName = engine.getBindingName(i);
                gLogInfo << "Dumping output tensor " << tensorName << ":" << std::endl;
                bufferManager.dumpBuffer(gLogInfo, tensorName);
            }
        }
    }
*/
   
cout<<"Yes dude"<<endl;
cudaStreamDestroy(stream);


  //CHECK(cudaFree(buffers[inputBindingIndex]));
cout<<"damn it"<<endl;
  // CHECK(cudaFree(buffers[outputBindingIndex]));
 cout<<"Before context"<<endl;
//    cudaStreamDestroy(stream);
  //  cudaEventDestroy(start);
   // cudaEventDestroy(end);
    context->destroy();
    //cudaDeviceReset();
  cout<<"We did reach here also"<<endl;
}



void launch_multistream(IExecutionContext* context,int inputBindingIndex,int outputBindingIndex, int outputBindingIndex1,Dims
outputDims, Dims outputDims1){

static const int INPUT_C = 3;
static const int INPUT_H = 416;
static const int INPUT_W = 416;

void* buffers[3];

/* int inputBindingIndex, outputBindingIndex,outputBindingIndex1;
  inputBindingIndex = engine.getBindingIndex("000_net");
  outputBindingIndex = engine.getBindingIndex("094_convolutional");
  outputBindingIndex1 = engine.getBindingIndex("106_convolutional");
*/


cout<<"The input binding index is"<<inputBindingIndex<<endl;
cout<<"The output binding index is "<<outputBindingIndex<<endl;




  if (inputBindingIndex < 0)
  {
    cout << "Invalid input name." << endl;
    return;
  }

  if (outputBindingIndex < 0)
  {
    cout << "Invalid output name." << endl;
    return;
  }


CHECK(cudaMalloc(&buffers[inputBindingIndex], gParams.batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));



// cout << "Preprocessing input..." << endl;
  cv::Mat image = cv::imread("/usr/src/tensorrt/samples/trtexec/sample.jpg", cv::IMREAD_COLOR);

  if (image.data == NULL)
  {
    cout << "Could not read image from file." << endl;
    return;
  }

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
  cv::resize(image, image, cv::Size(INPUT_W, INPUT_H));


float *inputDataHost, *outputDataHost;
size_t  numOutput,numOutput1;
  //numInput = numTensorElements(inputDims);
  numOutput = numTensorElements(outputDims);
  numOutput1= numTensorElements(outputDims1);
 // cout<<"The size is"<<numOutput<<"The other is"<<numOutput1<<endl;
 CHECK(cudaMalloc(&buffers[outputBindingIndex], numOutput * sizeof(float)));
 CHECK(cudaMalloc(&buffers[outputBindingIndex1], numOutput1 * sizeof(float)));


 //ouitputDataHost = (float*) malloc(numOutput * sizeof(float));
float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f };

//float* data = new float[1*size[0]*size[1]*size[2]];
//cout<<"The building information is"<<cv::getBuildInformation()<<endl;
//cv::setNumThreads(0);
//cv::setNumThreads(1);


float* data = new float[gParams.batchSize*INPUT_C*INPUT_H*INPUT_W];
//cout<<"after data"<<endl;
cv::Mat_<cv::Vec3f>::iterator it;
        unsigned volChl = 416*416;
        for (int c = 0; c < 3; ++c)
        {     
               
                cv::Mat_<cv::Vec3f>::const_iterator  it = image.begin<cv::Vec3f>();    //cv::Vec3f not working 
                // the color image to input should be in BGR order
               
                 for (unsigned j = 0; j < INPUT_H*INPUT_W; ++j)
                {       


                       //cout<<"Inside the main loop"<<endl;
                        //cout<<"Data is"<<data[j]<<endl;
                        //OpenCV read in frame as BGR format, by default, thus need only deduct the mean value
                        data[c*INPUT_W*INPUT_H + j] = float((*it)[c]) - pixelMean[c];
                       // it++;
                }

           }



cudaStream_t stream;
CHECK(cudaStreamCreate(&stream));



cudaStreamSynchronize(stream);
//bindings[0]=(void*)inputDataDevice;

///cout<<"The inputdatadevice is"<<inputDataDevice<<endl;
//out<<"The binding is"<<bindings[0]<<endl;
//bool var=context->execute(1,&bindings[0]);
constexpr int ITERATIONS = 1000;
cudaEvent_t start;
cudaEvent_t end;
 double totalTime = 0.0;
cudaEventCreate(&start);
cudaEventCreate(&end);



//cout << "execute\n";
cudaStreamSynchronize(stream);
size_t output_size = numOutput;//output_dim_.c() * output_dim_.h() * output_dim_.w();
std::vector<float> output(output_size);


size_t output_size1 = numOutput1;//output_dim_.c() * output_dim_.h() * output_dim_.w();
std::vector<float> output1(output_size1);
//cout<<"Before loop"<<endl;
cudaProfilerStart();
for(int i=0;i<ITERATIONS;i++){
mu.lock();
float elapsedTime;
//cout<<"Before event start"<<endl;
cudaEventCreate(&start);
cudaEventCreate(&end);
cudaError_t st = cudaEventRecord(start,stream);
if(st!=cudaSuccess){
cout<<"There is an error"<<endl;

}
//cout<<"Yes dude"<<endl;
//cudaProfilerStart();
 auto t_start = std::chrono::high_resolution_clock::now();
cudaError_t cudaStatus=cudaMemcpyAsync(buffers[inputBindingIndex], data, gParams.batchSize * INPUT_W * INPUT_H * INPUT_C * sizeof(float),
 cudaMemcpyHostToDevice,stream);

bool var=context->enqueue(gParams.batchSize, buffers, stream, nullptr);

 cudaStatus=cudaMemcpyAsync(output.data(), buffers[outputBindingIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost,
stream);
cudaStatus=cudaMemcpyAsync(output1.data(), buffers[outputBindingIndex1], output_size1 * sizeof(float),
 cudaMemcpyDeviceToHost,stream);

//cudaProfilerStop();
cudaStreamSynchronize(stream);
cudaEventRecord(end, stream);
cudaEventSynchronize(end);


cudaStreamSynchronize(stream);
cudaEventElapsedTime(&elapsedTime, start, end);
totalTime += elapsedTime;

mu.unlock();
//auto t_end = std::chrono::high_resolution_clock::now();
  //              float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
   // cout << "Execution done in " << ms << " ms\n";
//cout<<"The value is "<<var<<endl;
}

cudaProfilerStop();
cout << "Inference batch size " << gParams.batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS<<"ms"<<endl;

//cout<<"Yes dude"<<endl;
cudaStreamDestroy(stream);


  //CHECK(cudaFree(buffers[inputBindingIndex]));
//cout<<"damn it"<<endl;
  // CHECK(cudaFree(buffers[outputBindingIndex]));
 //cout<<"Before context"<<endl;
//    cudaStreamDestroy(stream);
  //  cudaEventDestroy(start);
   // cudaEventDestroy(end);
//    context->destroy();
    //cudaDeviceReset();
  //cout<<"We did reach here also"<<endl;





















//cudaStream_t stream;
//CHECK(cudaStreamCreate(&stream));
///cout<<"Using thread"<<endl;
 //CHECK(cudaStreamCreate(&stream));
//context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
//std::this_thread::sleep_for(std::chrono::milliseconds(200));
//cout<<"We have reached here dude"<<endl;
}

void doInference_now(ICudaEngine& engine){
 cout<<"Let us test ths"<<endl;
//context-> destroy();

}
void doInference_thread(ICudaEngine& engine){
    int i; 
    //thread tid;
 //   pthread_t tid; 
       IExecutionContext* context = engine.createExecutionContext();
     //  std::shared_ptr<ICudaEngine> emptyPtr{};
     //  std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &engine);
      // samplesCommon::BufferManager bufferManager(aliasPtr, gParams.batchSize);
      // std::vector<void*> buffers = bufferManager.getDeviceBindings();

    // Let us create three threads 
//    for (i = 0; i < 3; i++) 
       int inputBindingIndex, outputBindingIndex,outputBindingIndex1;
  inputBindingIndex = engine.getBindingIndex("000_net");
  outputBindingIndex = engine.getBindingIndex("016_convolutional");
  outputBindingIndex1 = engine.getBindingIndex("023_convolutional");
        Dims outputDims=engine.getBindingDimensions(outputBindingIndex);
       Dims outputDims1=engine.getBindingDimensions(outputBindingIndex1);

        thread tid(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
      /*  thread tid1(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid2(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid3(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);  
     /*   thread tid4(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);  
        thread tid5(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid6(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);       
        thread tid7(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
      /* thread tid8(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid9(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid10(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid11(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
      /*  thread tid12(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid13(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid14(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid15(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
      /* thread tid16(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid17(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid18(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid19(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
       /* thread tid20(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid21(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid22(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid23(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
       thread tid24(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid25(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid26(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid27(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
/*      thread tid28(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid29(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid30(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid31(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid32(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid33(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid34(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid35(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);

     thread tid36(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid37(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid38(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid39(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);

     thread tid40(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid41(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid42(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid43(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);


     thread tid44(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid45(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid46(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid47(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);


     thread tid48(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid49(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid50(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid51(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);


     thread tid52(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid53(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid54(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid55(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);


      thread tid56(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
  thread tid57(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
        thread tid58(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
     thread tid59(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
       
*/

//thread tid1(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);

 
       //thread tidother(launch_multistream,context,inputBindingIndex,outputBindingIndex,outputBindingIndex1,outputDims,outputDims1);
      // thread::id id1 = tid.get_id();
      //thread::id id2 = tid1.get_id();
      //thread::id id3 = tidother.get_id();
         tid.join();
       /*  tid1.join();
         tid2.join();
         tid3.join();
       /*  tid4.join();
         tid5.join();
         tid6.join();
         tid7.join();
       /*  tid8.join();
       tid9.join();
       tid10.join();
        tid11.join();
       /* tid12.join();
        tid13.join();
       tid14.join();
        tid15.join();
       /* tid16.join();
        tid17.join();
        tid18.join();
        tid19.join();
    /*     tid20.join();
        tid21.join();
        tid22.join();
        tid23.join();
       tid24.join();
        tid25.join();
        tid26.join();
        tid27.join();
  /*      tid28.join();
        tid29.join();
        tid30.join();
        tid31.join();
        tid32.join();
        tid33.join();
        tid34.join();
        tid35.join();
        
        tid36.join();
        tid37.join();
        tid38.join();
        tid39.join();

         tid40.join();
        tid41.join();
        tid42.join();
        tid43.join();


         tid44.join();
        tid45.join();
        tid46.join();
        tid47.join();

         tid48.join();
        tid49.join();
        tid50.join();
        tid51.join();


         tid52.join();
        tid53.join();
        tid54.join();
        tid55.join();


         tid56.join();
        tid57.join();
        tid58.join();
        tid59.join();
*/
/*
         tid32.join();
        tid33.join();
        tid34.join();
        tid35.join();
*/
       
    // tid.join();
       // tid1.join();
       // tid.join();
       // tid1.join();
       // tid.join();
       // tid1.join();

        //tidother.join();
     //  cout<<"Thread"<<id1<<"got executed"<<endl;
       //cout<<"Thread"<<id2<<"got executed"<<endl;
       //cout<<"Thread"<<id3<<"got executed"<<endl;

      //  pthread_create(&tid, NULL, &launch_multistream, NULL); 
    //    pthread_join(tid,NULL);
      // cout<<"We came back"<<endl;
      context->destroy();


}

void doInference_check(ICudaEngine& engine)
{
    IExecutionContext* context = engine.createExecutionContext();
    IExecutionContext* context1 = engine.createExecutionContext();

    // Use an aliasing shared_ptr since we don't want engine to be deleted when bufferManager goes out of scope.
    std::shared_ptr<ICudaEngine> emptyPtr{};
    std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &engine);
    samplesCommon::BufferManager bufferManager(aliasPtr, gParams.batchSize);
    std::vector<void*> buffers = bufferManager.getDeviceBindings();
     cout<<"Hey the Bufffers are"<<buffers[0]<<endl;

    cudaStream_t stream,stream1;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaStreamCreate(&stream1));

    cudaEvent_t start, end;
    unsigned int cudaEventFlags = gParams.useSpinWait ? cudaEventDefault : cudaEventBlockingSync;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventFlags));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventFlags));

    std::vector<float> times(gParams.avgRuns);
    for (int j = 0; j < gParams.iterations; j++)
    {
        float totalGpu{0}, totalHost{0}; // GPU and Host timers
        for (int i = 0; i < gParams.avgRuns; i++)
        {
            auto tStart = std::chrono::high_resolution_clock::now();
            cudaEventRecord(start, stream);
            context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
            context1->enqueue(gParams.batchSize, &buffers[0],stream1,nullptr);
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);

            auto tEnd = std::chrono::high_resolution_clock::now();
            totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[i] = ms;
            totalGpu += ms;
        }
        totalGpu /= gParams.avgRuns;
        totalHost /= gParams.avgRuns;
        gLogInfo << "Average over " << gParams.avgRuns << " runs is " << totalGpu << " ms (host walltime is " << totalHost
                 << " ms, " << static_cast<int>(gParams.pct) << "\% percentile time is " << percentile(gParams.pct, times) << ")." << std::endl;
    }

    if (gParams.dumpOutput)
    {
        bufferManager.copyOutputToHost();
        int nbBindings = engine.getNbBindings();
        for (int i = 0; i < nbBindings; i++)
        {
            if (!engine.bindingIsInput(i))
            {
                const char* tensorName = engine.getBindingName(i);
                gLogInfo << "Dumping output tensor " << tensorName << ":" << std::endl;
                bufferManager.dumpBuffer(gLogInfo, tensorName);
            }
        }
    }

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    context->destroy();
}



















static void printUsage()
{
    printf("\n");
    printf("Mandatory params:\n");
    printf("  --deploy=<file>          Caffe deploy file\n");
    printf("  OR --uff=<file>          UFF file\n");
    printf("  OR --onnx=<file>         ONNX Model file\n");
    printf("  OR --loadEngine=<file>   Load a saved engine\n");

    printf("\nMandatory params for UFF:\n");
    printf("  --uffInput=<name>,C,H,W Input blob name and its dimensions for UFF parser (can be specified multiple times)\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

    printf("\nMandatory params for Caffe:\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

    printf("\nOptional params:\n");
    printf("  --model=<file>          Caffe model file (default = no model, random weights used)\n");
    printf("  --batch=N               Set batch size (default = %d)\n", gParams.batchSize);
    printf("  --device=N              Set cuda device to N (default = %d)\n", gParams.device);
    printf("  --iterations=N          Run N iterations (default = %d)\n", gParams.iterations);
    printf("  --avgRuns=N             Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
    printf("  --percentile=P          For each iteration, report the percentile time at P percentage (0<=P<=100, with 0 representing min, and 100 representing max; default = %.1f%%)\n", gParams.pct);
    printf("  --workspace=N           Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
    printf("  --safe                  Only test the functionality available in safety restricted flows.\n");
    printf("  --fp16                  Run in fp16 mode (default = false). Permits 16-bit kernels\n");
    printf("  --int8                  Run in int8 mode (default = false). Currently no support for ONNX model.\n");
    printf("  --verbose               Use verbose logging (default = false)\n");
    printf("  --saveEngine=<file>     Save a serialized engine to file.\n");
    printf("  --loadEngine=<file>     Load a serialized engine from file.\n");
    printf("  --calib=<file>          Read INT8 calibration cache file.  Currently no support for ONNX model.\n");
    printf("  --useDLACore=N          Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.\n");
    printf("  --allowGPUFallback      If --useDLACore flag is present and if a layer can't run on DLA, then run on GPU. \n");
    printf("  --useSpinWait           Actively wait for work completion. This option may decrease multi-process synchronization time at the cost of additional CPU usage. (default = false)\n");
    printf("  --dumpOutput            Dump outputs at end of test. \n");
    printf("  -h, --help              Print usage\n");
    fflush(stdout);
}

bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

template<typename T>
bool parseAtoi(const char* arg, const char* name, T& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = static_cast<T>(atoi(arg + n + 3));
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
    return parseAtoi<int>(arg, name, value);
}

bool parseUnsigned(const char* arg, const char* name, unsigned int& value)
{
    return parseAtoi<unsigned int>(arg, name, value);
}

// parse a boolean option of the form --name, or optionally, -letter.
bool parseBool(const char* arg, const char* name, bool& value, char letter = '\0')
{
    bool match = arg[0] == '-' && ((arg[1] == '-' && !strcmp(arg + 2, name)) || (letter && arg[1] == letter && !arg[2]));
    if (match)
    {
        // Always report the long form of the option.
        gLogInfo << name << std::endl;
        value = true;
    }
    return match;
}

bool parseFloat(const char* arg, const char* name, float& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atof(arg + n + 3);
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool validateArgs()
{
    // UFF and Caffe files require output nodes to be specified.
    if ((!gParams.uffFile.empty() || !gParams.deployFile.empty()) && gParams.outputs.empty())
    {
        gLogError << "ERROR: At least one output must be specified." << std::endl;
        return false;
    }
    if (!gParams.uffFile.empty() && gParams.uffInputs.empty())
    {
        gLogError << "ERROR: At least one UFF input must be specified to run UFF models." << std::endl;
        return false;
    }
    if (!gParams.loadEngine.empty() && !gParams.saveEngine.empty())
    {
        gLogError << "ERROR: --saveEngine and --loadEngine cannot be specified at the same time." << std::endl;
        return false;
    }
    return true;
}

bool parseArgs(int argc, char* argv[])
{
    if (argc < 2)
    {
        printUsage();
        return false;
    }

    for (int j = 1; j < argc; j++)
    {
        if (parseString(argv[j], "model", gParams.modelFile)
            || parseString(argv[j], "deploy", gParams.deployFile))
        {
            continue;
        }
        if (parseString(argv[j], "saveEngine", gParams.saveEngine))
        {
            continue;
        }
        if (parseString(argv[j], "loadEngine", gParams.loadEngine))
        {
            continue;
        }
        if (parseString(argv[j], "engine", gParams.engine))
        {
            gLogError << "--engine has been deprecated. Please use --saveEngine and --loadEngine instead." << std::endl;
            return false;
        }
        if (parseString(argv[j], "uff", gParams.uffFile))
        {
            continue;
        }

        if (parseString(argv[j], "onnx", gParams.onnxModelFile))
        {
            continue;
        }

        if (parseString(argv[j], "calib", gParams.calibrationCache))
            continue;

        std::string input;
        if (parseString(argv[j], "input", input))
        {
            gLogWarning << "--input has been deprecated and ignored." << std::endl;
            continue;
        }

        std::string output;
        if (parseString(argv[j], "output", output))
        {
            gParams.outputs.push_back(output);
            continue;
        }

        std::string uffInput;
        if (parseString(argv[j], "uffInput", uffInput))
        {
            std::vector<std::string> uffInputStrs = split(uffInput, ',');
            if (uffInputStrs.size() != 4)
            {
                gLogError << "Invalid uffInput: " << uffInput << std::endl;
                return false;
            }

            gParams.uffInputs.push_back(std::make_pair(uffInputStrs[0], Dims3(atoi(uffInputStrs[1].c_str()), atoi(uffInputStrs[2].c_str()), atoi(uffInputStrs[3].c_str()))));
            continue;
        }

        if (parseInt(argv[j], "batch", gParams.batchSize)
            || parseInt(argv[j], "iterations", gParams.iterations)
            || parseInt(argv[j], "avgRuns", gParams.avgRuns)
            || parseInt(argv[j], "device", gParams.device)
            || parseInt(argv[j], "workspace", gParams.workspaceSize)
            || parseInt(argv[j], "useDLACore", gParams.useDLACore))
            continue;

        if (parseFloat(argv[j], "percentile", gParams.pct))
            continue;

        if (parseBool(argv[j], "safe", gParams.safeMode)
            || parseBool(argv[j], "fp16", gParams.fp16)
            || parseBool(argv[j], "int8", gParams.int8)
            || parseBool(argv[j], "verbose", gParams.verbose)
            || parseBool(argv[j], "allowGPUFallback", gParams.allowGPUFallback)
            || parseBool(argv[j], "useSpinWait", gParams.useSpinWait)
            || parseBool(argv[j], "dumpOutput", gParams.dumpOutput)
            || parseBool(argv[j], "help", gParams.help, 'h'))
            continue;

        gLogError << "Unknown argument: " << argv[j] << std::endl;
        return false;
    }

    return validateArgs();
}

static ICudaEngine* createEngine()
{
	cout<<"In the engine function"<<endl;
    ICudaEngine* engine;
    // load directly from serialized engine file if deploy not specified
    if (!gParams.loadEngine.empty())
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(gParams.loadEngine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        IRuntime* infer = createInferRuntime(sample::gLogger.getTRTLogger());
        if (gParams.useDLACore >= 0)
        { 
            cout<<"The engine will be built with DLA thing"<<endl;
            infer->setDLACore(gParams.useDLACore);
        }

        engine = infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
        gLogInfo << gParams.loadEngine << " has been successfully loaded." << std::endl;

        infer->destroy();
        return engine;
    }

    if ((!gParams.deployFile.empty()) || (!gParams.uffFile.empty()) || (!gParams.onnxModelFile.empty()))
    {

        if (!gParams.uffFile.empty())
        {
            engine = uffToTRTModel();
        }
        else if (!gParams.onnxModelFile.empty())
        {
           cout<<"We entered in onnx model"<<endl;
           engine = onnxToTRTModel();
        }
        else
        {
            //engine = caffeToTRTModel();
        }

        if (!engine)
        {
            gLogError << "Engine could not be created" << std::endl;
            return nullptr;
        }

        if (!gParams.saveEngine.empty())
        {
            std::ofstream p(gParams.saveEngine, std::ios::binary);
            if (!p)
            {
                gLogError << "could not open plan output file" << std::endl;
                return nullptr;
            }
            IHostMemory* ptr = engine->serialize();
            if (ptr == nullptr)
            {
                gLogError << "could not serialize engine." << std::endl;
                return nullptr;
            }
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
            gLogInfo << "Engine has been successfully saved to " << gParams.saveEngine << std::endl;
        }
        return engine;
    }

    // complain about empty deploy file
    gLogError << "Deploy file not specified" << std::endl;
    return nullptr;
}

int main(int argc, char** argv)
{
    // create a TensorRT model from the caffe/uff/onnx model and serialize it to a stream

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    sample::gLogger.reportTestStart(sampleTest);

    if (!parseArgs(argc, argv))
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    if (gParams.help)
    {
        printUsage();
        return sample::gLogger.reportPass(sampleTest);
    }

    if (gParams.verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }

    cudaSetDevice(gParams.device);

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
	//cudaProfilerStart();


     auto t_start = std::chrono::high_resolution_clock::now();
      ICudaEngine*  engine = createEngine();
      auto t_end = std::chrono::high_resolution_clock::now();
          
       //cudaProfilerStop();

      float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    //  cout << "Execution done in " << ms << " ms\n";
    if (!engine)
    {
        gLogError << "Engine could not be created" << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    if (gParams.uffFile.empty() && gParams.onnxModelFile.empty())
    {
        nvcaffeparser1::shutdownProtobufLibrary();
    }
    else if (gParams.deployFile.empty() && gParams.onnxModelFile.empty())
    {
        nvuffparser::shutdownProtobufLibrary();
    }
   // pthread_t thread_id;
   // printf("Before Thread\n");
   // pthread_create(&thread_id, NULL,doInference, *engine);
   // pthread_join(thread_id, NULL);
 //cudaProfilerStart();
  //cout<<"Hey dude, How is it going"<<endl;    
//doInference_thread(*engine); 

//cudaStream_t stream;
//CHECK(cudaStreamCreate(&stream));
//cudaEvent_t start;
//cudaEvent_t end;

//cudaStreamSynchronize(stream);

 //double totalTime = 0.0;
//cudaEventCreate(&start);
//cudaEventCreate(&end);
//float elapsedTime;
//cudaProfilerStart();
//cudaError_t st = cudaEventRecord(start,stream);
//cudaProfilerStart();
doInference_thread(*engine);
//doInference_now(*engine);
//cudaProfilerStop();
//cudaProfilerStop();
//cudaStreamSynchronize(stream);

//cudaEventRecord(end,stream);
//cudaEventElapsedTime(&elapsedTime, start, end);
//cout<<"The elapsed time is "<<elapsedTime<<endl;
//auto t_end1 = std::chrono::high_resolution_clock::now();
//float milli = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
//cout << "The inferencing is done dude in " << milli << " ms\n";


 //    doInference(*engine);
   // thread th1(doInference, ICudaEngine* engine);
   // th1.join();
    cout<<"We came back after inference"<<endl;
    engine->destroy();

    return sample::gLogger.reportPass(sampleTest);
}

