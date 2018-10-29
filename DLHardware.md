# Overview 
WIP of recent options, this table is by no means complete. Contributions welcome!

| Architecture | Chip/Series         | Company   | Vendors                 | Model(s)                                                                                                                                                           | Cost                | Backend                                              | Rec. Frameworks                       | Platform           | Inference | Training |
|--------------|---------------------|-----------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|------------------------------------------------------|---------------------------------------|--------------------|-----------|----------|
| CPU          | Coffee Lake Refresh | Intel     | Intel                   | [9th Gen](https://www.anandtech.com/show/13400/intel-9th-gen-core-i9-9900k-i7-9700k-i5-9600k-review)                                                               | vary                | MKL                                                  | all major DL frameworks               | Desktop            | y         | y        |
| CPU          | Coffee Lake         | Intel     | Intel                   | [Xeon](https://en.wikipedia.org/wiki/List_of_Intel_Xeon_microprocessors#Coffee_Lake-based_Xeons)                                                                   | vary                | MKL                                                  | all major DL frameworks               | typically Cloud    | y         | y        |
| GPU          | Turing              | NVIDIA    | NVIDIA + Partners       | [RTX 20 Series](https://en.wikipedia.org/wiki/GeForce_20_series)                                                                                                   | vary                | [CUDA/cuDNN                                          | all major DL frameworks               | Desktop            | y         | y        |
| GPU          | Volta               | NVIDIA    | NVIDIA HPC Partner      | [Tesla V100](https://www.nvidia.com/en-us/data-center/tesla-v100/)                                                                                                 | vary                | CUDA/cuDNN                                           | all major DL frameworks               | Cloud              | y         | y        |
| ARM + GPU    | Carmel + Volta      | NVIDIA    | NVIDIA                  | [Jetson Xavier](https://developer.nvidia.com/embedded/buy/jetson-xavier-devkit)                                                                                    | $2,499              | cuDNN                                                | TensorRT                              | Dev. Board         | y         | n        |
| GPU          | Vega                | AMD       | AMD + Partners          | [RX Vega](https://en.wikipedia.org/wiki/AMD_RX_Vega_series)                                                                                                        | vary                | ROCm                                                 | TensorFlow, Caffe                     | Dekstop, Cloud     | y         | y        |
| ASIC         | TPUv2               | Google    | Google Compute Cloud    | [TPUv2](https://cloud.google.com/tpu/)                                                                                                                             | $4.5/h              | -                                                    | TensorFlow                            | Cloud              | y         | y        |
| ARM+ASIC     | TPU                 | Google    | Google                  | [Edge TPU Dev Board and Accelerator](https://aiyprojects.withgoogle.com/edge-tpu)                                                                                  | NA                  | NNAPI                                                | TensorFlow Lite                       | Dev. Board         | y         | n        |
| ARM          | BCM2837B0           | Broadcom  | Raspberry Pi Foundation | [Raspberry Pi 3+](https://www.raspberrypi.org)                                                                                                                     | $35                 | ARMComputeLib, LLVM, NNPACK, Openblas, NNAPI         | TVM, PyTorch, Caffe2, TensorFlow Lite | Dev. Board         | y         | n        |
| ARM          | RK3999              | Rockchip  | Pine64, Vamrs, Firefly  | [ROCKPro64](https://www.pine64.org/?page_id=61454),[Rock96](https://www.96boards.org/product/rock960/),[Firefly-RK3399](http://shop.t-firefly.com/goods.php?id=45) | $59.99 - $259.99    | OpenCL, ARMComputeLib, LLVM, NNPACK, Openblas, NNAPI | TVM, PyTorch, Caffe2, TensorFlow Lite | Dev. Board         | y         | n        |
| ARM+ASIC     | RK3999Pro           | Rockchip  | Pine64?, Vamrs?         | NA                                                                                                                                                                 | NA                  | NA                                                   | NA                                    | Dev. Board         | NA        | NA       |
| ARM+ASIC     | Kirin970            | HiSilicon | Huawai                  | [HiKey970](https://www.96boards.org/product/hikey970/), [Honor Play](https://www.hihonor.com/uk/product/10044248721055.html)                                       | vary                | HiAI                                          | HiAI, TensorFlow Lite, Caffe          | Mobile, Dev. Board | y         | n        |
| ARM+ASIC     | Kirin980            | HiSilicon | Huawai                  | [Mate 20 Pro](https://consumer.huawei.com/en/phones/mate20-pro/), [Mate 20](https://consumer.huawei.com/en/phones/mate20/)                                         | £899-               | HiAI                                          | HiAI, TensorFlow Lite, Caffe          | Mobile             | y         | n        |
| ARM+ASIC     | A12 Bionic          | Apple     | Apple                   | [iPhone XS](https://www.apple.com/uk/shop/buy-iphone/iphone-xs), [iPhone XR](https://www.apple.com/uk/iphone-xr/)                                                  | $749-$1449          | Metal2, BNNS, NNPACK                                  | CoreML, Caffe2, TensorFlow Lite       | Mobile             | y         | n        |
| FPGA         | ARM A9, Artix-7     | Xilinx    | Xilinx                  | [PYNQ-Z1](https://www.xilinx.com/support/university/boards-portfolio/xup-boards/XUPPYNQ.html#overview)                                                             | $199                | VTA                                                  | TVM                                   | Dev. Board         | y         | n        |
| ARM+FPGA     | UltraScale+ MPSoC   | Xilinx    | Xilinx                  | [Ultra-96](https://www.96boards.org/product/ultra96/)                                                                                                              | $249                | -                                                    | TVM announced future support          | Dev. Board         | y         | n        |
| FPGA         | XCU200, XUC250      | Xilinx    | Xilinx                  | [Alveo U200](https://www.xilinx.com/products/boards-and-kits/alveo/u200.html),[Alveo U250](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html)        | $8,995 - $12,995,   | xDNN                                                 | xfDNN                                 | Server             | y         | n        |
| FPGA         | UltraScale+ VU9P    | Xilinx    | Xilinx, Amazon AWS      | [EC2 F1](https://aws.amazon.com/ec2/instance-types/f1/)                                                                                                            | $1.65/h - $13.20/hr | ZEBRA                                                | Caffe                                 | Cloud              | y         | n        |


# Resources

## Backends:
* [MKL](https://software.intel.com/en-us/mkl)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [ROCm](https://rocm.github.io)
* [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks/)
* [OpenCL](https://www.khronos.org/opencl/)
* [Arm Compute Library](https://developer.arm.com/technologies/compute-library)
* [NNPACK](https://github.com/Maratyszcza/NNPACK)
* [LLVM](http://llvm.org)
* [Openblas](https://www.openblas.net)
* [HiAI](https://developer.huawei.com/consumer/en/devservice/doc/2020314)
* [Metal 2](https://developer.apple.com/metal/)
* [BNNS](https://developer.apple.com/documentation/accelerate/bnns)
* [VTA](https://tvm.ai/2018/07/12/vta-release-announcement.html) 
* [xDNN](https://github.com/Xilinx/ml-suite)
* [ZEBRA](https://aws.amazon.com/marketplace/pp/B073SHB43M)

## Frameworks
* [ONNX](https://onnx.ai) (Open Neural Network Exchange Format)
* [PyTorch](https://pytorch.org) (includes Caffe2)
* [TVM](https://tvm.ai)
* [CoreML](https://developer.apple.com/documentation/coreml)
* [xfDNN](https://github.com/Xilinx/ml-suite)
* [TensorFlow](https://www.tensorflow.org)
* [TensorFlow Lite](https://www.tensorflow.org/lite/)
* [MXNet](https://mxnet.incubator.apache.org)
* [Caffe](http://caffe.berkeleyvision.org)
* [TensorRT](https://developer.nvidia.com/tensorrt)



## Others
* [DAWNBench](https://dawn.cs.stanford.edu/benchmark/): An End-to-End Deep Learning Benchmark and Competition by Stanford DAWN
* [Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](http://timdettmers.com/2018/08/21/which-gpu-for-deep-learning/): Blog post by Tim Dettmers (@TimDettmers)
* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799): TVM paper on arXiv.org
* [Machine Learning for iOS](https://github.com/alexsosn/iOS_ML): list of resources for iOS developers by Alex Sosnovshchenko (@alexsosn)
* [AI Chip List](https://github.com/basicmi/AI-Chip-List): AI Chip List by Shan Tang (@basicmi)
* [Net Runner](https://github.com/doc-ai/net-runner-ios): iOS  environment for running, measuring, and evaluating computer vision machine learning models on device by doc-ai
* [AIMark](https://itunes.apple.com/us/app/aimark/id1377968254?mt=8): iOS app benchmark app for popular image classification networks (Inception V3, ResNet34, VGG16) by Master Lu

# Examples: 
* [Transfering a model from PyTorch to Caffe2 and mobile using ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html#sphx-glr-download-advanced-super-resolution-with-caffe2-py)
* [Deploy a trained PyTorch model on the ROCKPro64 Development Board](link_file_in_same_repository)
* [Compile  deep learning models in TVM](https://docs.tvm.ai/tutorials/#compile-deep-learning-models)

---- here starts the new repository file --- 

##Trained PyTorch model on the ROCKPro64 Development Board

### Option 1: Install PyTorch on the device

### Option2 : PyTorch -> ONNX -> TVM 