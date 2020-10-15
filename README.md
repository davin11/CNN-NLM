# CNN-NLM : Nonlocal CNN SAR Image Despeckling
[Nonlocal CNN SAR Image Despeckling](https://www.mdpi.com/2072-4292/12/6/1006) is 
a method for SAR image despeckling which performs nonlocal filtering with a deep learning engine.

## Team members
 Davide Cozzolino (davide.cozzolino@unina.it);
 Luisa Verdoliva  (verdoliv@.unina.it);
 Giuseppe Scarpa  (giscarpa@.unina.it);
 Giovanni Poggi   (poggi@.unina.it).
 
## License
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 

## Prerequisits
All the functions and scripts were tested on Python 3.6, PyTorch 0.4.1 and Cuda 9.2,
the operation is not guaranteed with other configurations.
The command to create the CONDA environment: 
```
conda env create -n env_cnn_nlm -f environment.yml
```

The command to anctivate the CONDA environment:
```
conda activate env_cnn_nlm
```

The command to install PyInn: 
```
pip install git+https://github.com/szagoruyko/pyinn.git@master
```

The commands to install matmul_cuda:
```
svn export https://github.com/visinf/n3net.git/trunk/lib
sed -i 's/extension.h/torch.h/g' lib/matmul.cpp
cd lib; python setup.py install
```

Please download the datasets using the provided script:
```
bash download_sets.sh
python generate_noisy_synthetics.py
```

## Usage

### Demo
Use `demo_sync.py` to execute a demo for the network CNN-NLM on synthetic data.
coming soon: `demo_real.py`.

### Training and Testing
The command to train the network CNN-NLM on synthetic data:

```
CUDA_VISIBLE_DEVICES=0 python experiment_nlmcnn.py --exp_name new_train
```

The command to test the network CNN-NLM on synthetic data:

```
CUDA_VISIBLE_DEVICES=0 python experiment_nlmcnn.py --eval --eval_epoch 50 --exp_name new_train
```

The script `python experiment_sarcnn17.py` is the implementation in Python/Torch of the paper "SAR image despeckling through convolutional neural networks". 

NOTE: the SSIM of the paper is little different because it was computed using Matlab instead of Python. 
