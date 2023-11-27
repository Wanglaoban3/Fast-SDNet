# Fast-SDNet: An Extremely Lightweight Network For Surface Defect Segmentation on Edge Platform
In this study, we propose a highly lightweight network called Fast-SDNet for defect segmentation on edge platforms. The model code will be made publicly available upon acceptance of the paper.

## Python >= 3.6
PyTorch >= 1.1.0
onnx, onnxruntime, tqdm, tensorboardX

## Code usage
Once you have downloaded the dataset, you can start training by modifying the `dataset_root_path` parameter in the `main.py` file.

```bash
python main.py --model u_net --benchmark carpet --dataset_root_path [your path] --mode total-sup --batch_size 6 --base_lr 0.01 --epochs 100
```

 
## Acknowledgment
Thank you to all the individuals who have supported this project.

## Contact
For any issue please contact me at wangyi@s.upc.edu.cn
