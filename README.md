1.创建虚拟环境
python -m venv venv
2.激活虚拟环境
cmd venv\Scripts\activate powershell venv\Scripts\Activate.ps1 退出虚拟环境 deactivate
3.安装依赖
pip install --upgrade pip
pip install torch torchvision
pip install onnx

pip install onnxscript
pip install --upgrade onnx onnxscript
