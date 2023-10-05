# OpenGANForensics
Official PyTorch implementation of
["**Open Set Classification of GAN-based Image Manipulations via a ViT-based Hybrid Architecture**"](https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Wang_Open_Set_Classification_of_GAN-Based_Image_Manipulations_via_a_ViT-Based_CVPRW_2023_paper.pdf). 


## 1. Requirements
### Environments
Before running the code, please configure your env following the requirement file.

### Datasets
We collected our own dataset using the code and models released by [PTI](https://github.com/danielroich/PTI).

## 2. Training and testing
### We provide two networks for training. The dataset load function can be replaced by the users based on their own tasks.

To train the models in paper for classification and localization, run this command:
```train
python main.py --save_models ./path_to_save_model --model resnet50 --loc --nodown
```
Note: if loc and nodown are activated, the network is exactly the one we used in our work. Without --loc, it will be resnet50 + Vit network.

For open set test, simply run the command:
```Open set test
python test_osr.py --weights_path ./path_to_model --pretrain --loc --nodown
```
### Pretrained Model

You can find one of the pretrained models [here]([https://drive.google.com/file/d/1UJ8dhiMcbebMv6fHAVeGg2LOOrvrt1qC/view?usp=drive_link](https://drive.google.com/drive/folders/1tgVPNL4lCVYUkT1OyYhGDmz0lOu75EHW?usp=drive_link))
## Citation
- If you find our work or the code useful, please consider cite our paper using:
```bibtex
@inproceedings{wang2023open,
  title={Open Set Classification of GAN-based Image Manipulations via a ViT-based Hybrid Architecture},
  author={Wang, Jun and Alamayreh, Omran and Tondi, Benedetta and Barni, Mauro},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={953--962},
  year={2023}
}
```

## Contact
- If you find a problem with the code please contact me
