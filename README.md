Project done in CSE4007 2024
This project is about defending from PGD adversarial attacks by focusing on logits
The idea is based on the paper by Liao et al(2018) and could be found as below.
<Liao, F., Liang, M., Dong, Y., Pang, T., Hu, X., & Zhu, J. (2018). Defense against adversarial attacks using high-level representation guided denoiser. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1778-1787)>

I implemented denoiser to denoise adversarial noise and then used classifier
Throughout the project, Resnet50 was used as a baseline model.

### Descriptions for Files ###
- Adversarial_Logit_Pairing: demo of ALP 
- JointTraining: without pretrained CNN, jointly train CNN and denoiser
- Denoiser_PretrainedCNN: use vanilla training for pre-trained CNN, later to be frozen while training denoiser
- Denoiser_PretrainedALP: use ALP to obtain pre-trained CNN, later to be frozen while training denoiser

### Notes ###
1. Attacks can be imported from https://github.com/Harry24k/adversarial-attacks-pytorch.git
2. class DenoiseLoss, class Net are modified from https://github.com/lfz/Guided-Denoise.git
