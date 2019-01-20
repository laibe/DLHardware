# Overview 
Overview of some recent hardware platforms and their supported frameworks, this table is by no means complete. Contributions welcome!

| Architecture | Chip/Series                 | Company        | Vendors                 | Model(s)                                                                                                                                                              | Backend                                              | Rec. Frameworks                       | Platform           | Inference | Training |
|--------------|-----------------------------|----------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|---------------------------------------|--------------------|-----------|----------|
| CPU          | Coffee Lake Refresh         | Intel          | Intel                   | [9th Gen](https://www.anandtech.com/show/13400/intel-9th-gen-core-i9-9900k-i7-9700k-i5-9600k-review)                                                                  | MKL                                                  | all major DL frameworks               | Desktop            | y         | y        |
| CPU          | Coffee Lake                 | Intel          | Intel                   | [Xeon](https://en.wikipedia.org/wiki/List_of_Intel_Xeon_microprocessors#Coffee_Lake-based_Xeons)                                                                      | MKL                                                  | all major DL frameworks               | typically Cloud    | y         | y        |
| GPU          | Turing                      | NVIDIA         | NVIDIA + Partners       | [RTX 20 Series](https://en.wikipedia.org/wiki/GeForce_20_series)                                                                                                      | CUDA/cuDNN                                           | all major DL frameworks               | Desktop            | y         | y        |
| GPU          | Volta                       | NVIDIA         | NVIDIA HPC Partner      | [Tesla V100](https://www.nvidia.com/en-us/data-center/tesla-v100/)                                                                                                    | CUDA/cuDNN                                           | all major DL frameworks               | Cloud              | y         | y        |
| ARM + GPU    | Carmel + Volta              | NVIDIA         | NVIDIA                  | [Jetson Xavier](https://developer.nvidia.com/embedded/buy/jetson-xavier-devkit)                                                                                       | cuDNN                                                | TensorRT                              | Dev. Board         | y         | n        |
| GPU          | Vega                        | AMD            | AMD + Partners          | [RX Vega](https://en.wikipedia.org/wiki/AMD_RX_Vega_series)                                                                                                           | ROCm                                                 | TensorFlow, Caffe                     | Dekstop, Cloud     | y         | y        |
| ASIC         | TPUv2                       | Google         | Google Compute Cloud    | [TPUv2](https://cloud.google.com/tpu/)                                                                                                                                | -                                                    | TensorFlow                            | Cloud              | y         | y        |
| ARM+ASIC     | TPU                         | Google         | Google                  | [Edge TPU Dev Board and Accelerator](https://aiyprojects.withgoogle.com/edge-tpu) (NA)                                                                                | NNAPI                                                | TensorFlow Lite                       | Dev. Board         | y         | n        |
| ARM          | BCM2837B0                   | Broadcom       | Raspberry Pi Foundation | [Raspberry Pi 3+](https://www.raspberrypi.org)                                                                                                                        | ARMComputeLib, LLVM, NNPACK, Openblas, NNAPI         | TVM, PyTorch, Caffe2, TensorFlow Lite | Dev. Board         | y         | n        |
| ARM+ASIC     | BM1808                      | Bitmain/Sophon | Bitmain                 | [Neural Network Stick, Neural Network Module, Edge TPU Dev. Board](https://www.sophon.ai/site/index.html)                                                             | BMNet                                                | Caffe, ONNX, Tensorflow, Pytorch      | Dev. Board, Stick  | y         | n        |
| ARM          | RK3399                      | Rockchip       | Pine64, Vamrs, Firefly  | [ROCKPro64](https://www.pine64.org/?page_id=61454), [Rock960](https://www.96boards.org/product/rock960/), [Firefly-RK3399](http://shop.t-firefly.com/goods.php?id=45) | OpenCL, ARMComputeLib, LLVM, NNPACK, Openblas, NNAPI | TVM, PyTorch, Caffe2, TensorFlow Lite | Dev. Board         | y         | n        |
| ARM+ASIC     | RK1808 (formerly RK3399Pro) | Rockchip       | Pine64?, Vamrs?         | NA                                                                                                                                                                    | NA                                                   | NA                                    | Dev. Board         | NA        | NA       |
| ARM+ASIC     | Kirin970                    | HiSilicon      | Huawai                  | [HiKey970](https://www.96boards.org/product/hikey970/), [Honor Play](https://www.hihonor.com/uk/product/10044248721055.html)                                          | HiAI (for NPU), NNAPI, NNPACK, LLVM                  | HiAI, TensorFlow Lite, Caffe          | Mobile, Dev. Board | y         | n        |
| ARM+ASIC     | Kirin980                    | HiSilicon      | Huawai                  | [Mate 20 Pro](https://consumer.huawei.com/en/phones/mate20-pro/), [Mate 20](https://consumer.huawei.com/en/phones/mate20/)                                            | HiAI (for NPU), NNAPI, NNPACK,LLVM                   | HiAI, TensorFlow Lite, Caffe          | Mobile             | y         | n        |
| ARM+ASIC     | A12 Bionic                  | Apple          | Apple                   | [iPhone XS](https://www.apple.com/uk/shop/buy-iphone/iphone-xs), [iPhone XR](https://www.apple.com/uk/iphone-xr/)                                                     | Metal2, BNNS, NNPACK, LLVM                           | CoreML, Caffe2, TensorFlow Lite       | Mobile             | y         | n        |
| FPGA         | ARM A9, Artix-7             | Xilinx         | Xilinx                  | [PYNQ-Z1](https://www.xilinx.com/support/university/boards-portfolio/xup-boards/XUPPYNQ.html#overview)                                                                | VTA                                                  | TVM                                   | Dev. Board         | y         | n        |
| ARM+FPGA     | UltraScale+ MPSoC           | Xilinx         | Xilinx                  | [Ultra-96](https://www.96boards.org/product/ultra96/)                                                                                                                 | -                                                    | TVM (on roadmap)                      | Dev. Board         | y         | n        |
| FPGA         | XCU200, XUC250              | Xilinx         | Xilinx                  | [Alveo U200](https://www.xilinx.com/products/boards-and-kits/alveo/u200.html), [Alveo U250](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html)          | xDNN                                                 | xfDNN                                 | Server             | y         | n        |
| FPGA         | UltraScale+ VU9P            | Xilinx         | Xilinx, Amazon AWS      | [EC2 F1](https://aws.amazon.com/ec2/instance-types/f1/)                                                                                                               | ZEBRA                                                | Caffe, TVM (on roadmap)               | Cloud              | y         |          |

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
* [TVM](https://tvm.ai), [TVM 0.5 Roadmap](https://github.com/dmlc/tvm/issues/1596) 
* [CoreML](https://developer.apple.com/documentation/coreml)
* [xfDNN](https://github.com/Xilinx/ml-suite)
* [TensorFlow](https://www.tensorflow.org)
* [TensorFlow Lite](https://www.tensorflow.org/lite/)
* [MXNet](https://mxnet.incubator.apache.org)
* [Caffe](http://caffe.berkeleyvision.org)
* [TensorRT](https://developer.nvidia.com/tensorrt)



## Others

* [PyTorch 1.0: Bringing research and production together](https://cdn.oreillystatic.com/en/assets/1/event/286/PyTorch%201_0_%20Bringing%20research%20and%20production%20together%20Presentation.pdf): Slides by Dmytro Dzhulgakov 
* [DAWNBench](https://dawn.cs.stanford.edu/benchmark/): An End-to-End Deep Learning Benchmark and Competition by Stanford DAWN
* [Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](http://timdettmers.com/2018/08/21/which-gpu-for-deep-learning/): Blog post by Tim Dettmers 
* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799): TVM paper by Chen et al.
* [Machine Learning for iOS](https://github.com/alexsosn/iOS_ML): list of resources for iOS developers by Alex Sosnovshchenko
* [AI Chip List](https://github.com/basicmi/AI-Chip-List): AI Chip List by Shan Tang
* [Net Runner](https://github.com/doc-ai/net-runner-ios): iOS  environment for running, measuring, and evaluating computer vision machine learning models on device by doc-ai
* [AIMark](https://itunes.apple.com/us/app/aimark/id1377968254?mt=8): iOS app benchmark app for popular image classification networks (Inception V3, ResNet34, VGG16) by Master Lu

# Tutorials: 
## External Tutorials
* [Transfering a model from PyTorch to Caffe2 and mobile using ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html#sphx-glr-download-advanced-super-resolution-with-caffe2-py): Official PyTorch Tutorial
* [Compile  deep learning models in TVM](https://docs.tvm.ai/tutorials/#compile-deep-learning-models): Official TVM Tutorials

## Deploy a trained PyTorch model on the ROCKPro64

### Option 1: Install PyTorch on the device
```bash
sudo apt-get install python3-dev python3-setuptools python3-numpy

# Pillow only necessary for computer vision applications
sudo apt-get update
sudo apt-get install libjpeg-turbo8-dev zlib1g-dev libtiff5-dev
sudo pip3 install Pillow

# build PyTorch Master from source 
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export NO_CUDA=1
sudo python3 setup.py install

# Torchvision 
git clone https://github.com/pytorch/vision.git
cd vision
sudo python3 setup.py install

```

### Option 2 : PyTorch -> ONNX -> TVM 
#### Host machine Setup (macOS)
1.Ensure cmake, [XCode](https://developer.apple.com/xcode/), [LLVM](http://releases.llvm.org/download.html) and [Anaconda](https://www.anaconda.com/download/#macos) are installed. It is recommended to install cmake via [brew](https://brew.sh). 

2.Install the latest PyTorch nighly build (required for working ONNX export)
```bash
# Recommended to create a conda env
conda create --name pytorch_tvm_p36 python=3.6;
source activate pytorch_tvm_p36;
# see https://pytorch.org for more details
conda install pytorch-nightly -c pytorch
```  
Optional: Install torchvision:
```bash
git clone git@github.com:pytorch/vision.git;
cd vision;
sudo python3 setup.py install;
```
3.Install latest ONNX from source
```bash
conda install -c conda-forge protobuf
pip install onnx
```


3.Follow below steps to Install TVM from source (largely adapted from the [official TVM tutorial](https://docs.tvm.ai/install/from_source.html) )
```bash
git clone --recursive https://github.com/dmlc/tvm
mkdir build
cp cmake/config.cmake build
```  
modify `cmake.config` and set the LLVM path: `set(USE_LLVM /path/to/your/llvm/bin/llvm-config)`   
```bash
# build
cd build
cmake ..
make -j4
```  
Install the Python Package:
```bash
# TVM Python
cd /path/to/tvm;
export TVM_HOME=/path/to/tvm;
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH};
export MACOSX_DEPLOYMENT_TARGET=10.9;
cd python; python3 setup.py install; cd ..
cd topi/python; python3 setup.py install; cd ../..
cd nnvm/python; python3 setup.py install; cd ../..
```

#### Target device setup (ROCKPro64)  

Install OpenCL driver (only required if model should run on the Mali GPU)
```bash
sudo apt-get install libmali-rk-midgard-t86x-r14p0
sudo apt-get install opencl-headers
```  
Install TVM
```bash
git clone --recursive https://github.com/dmlc/tvm
cd tvm
cp cmake/config.cmake .
sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake
make runtime -j4
```
add `export PYTHONPATH=$PYTHONPATH:/path/to/tvm/python` to `~.bashrc`

### PyTorch -> ONNX (on Host)
```python
# minimal example code to load a PyTorch model and export it to onnx'
import onnx
import onnx.utils
import torch.nn as nn
import torch

state_dict = torch.load('/path/to/PyTorch_state_dict.pth',map_location='cpu')
model.load_state_dict(state_dict,strict=True)
model.eval()

# initiate random input for tracing (has to have same dimension as the model was trained with)
x = torch.randn(1, 3, 500, 500, requires_grad=True)
# export
out_path = '/output/path/model.onnx'
torch_out = torch.onnx.export(model,x,out_path,export_params=True,verbose=True)
# verify and polish
model = onnx.load(str(out_path))
onnx.checker.check_model(model)
polished_model = onnx.utils.polish_model(model)
onnx.checker.check_model(polished_model)
onnx.save(polished_model,'/output/path/polishedmodel.onnx')
```

#### ONNX->TVM 
This is the tricky part where we compile the model locally and deploy it to the target device via RPC. A good starting points are the official tutorials: 
* [Deploy model on Mali GPU](https://docs.tvm.ai/tutorials/nnvm/deploy_model_on_mali_gpu.html) 
* [Deploy Model on Raspberry Pi](https://docs.tvm.ai/tutorials/nnvm/deploy_model_on_rasp.html). 

The repository for the M2U-Net paper includes a script that compiles the model for deployment on a ROCKPro64 and benchmarks the inference times: [benchmark_tvm_arm.py](). (not yet available)