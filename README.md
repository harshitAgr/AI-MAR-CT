# AI-MAR-CT


 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

 Collection of deep learning-based metal artifact reduction (MAR) articles for CT/CBCT Imaging

 **PND-Net: Physics-inspired Non-local Dual-domain Network for Metal Artifact Reduction**  <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
 *J. Xia et al.* \
 **Summary**\
 The authors propose to use three networks. One is called non-local sinogram decomposition network (NSD-Net) to acquire the weighted artifact component in sinogram domain. The second network is an image restoration network (IR-Net) to reduce the residual and secondary artifacts in the image domain. The third trainable fusion network (F-Net) in the artifact synthesis path is used for unpaired learning. The method is compared with linear interpolation, NMAR, DuDoNet, DuDoNet++, InDuDoNet+, ADN, and U-DuDoNet.\
 IEEE TMI. [[doi](https://ieeexplore.ieee.org/document/10404006)] [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/Ballbo5354/PND-Net/tree/main)


**Dual Domain Diffusion Guidance for 3D CBCT Metal Artifact Reduction**  <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*Y. Choi et al.* \
**Summary** \
The authors propose to use two 2D diffusion models in image domain to synthesize metal artifacts and metal free images, respectively. The combination of metal artifact and metal free images is forward projected and a guidance from the error in the projection domain is used. The results are compared with InDuDoNet+, ACDNet, FEL, and Blind DPS methods.  \
WACV 2024. [[doi](https://openaccess.thecvf.com/content/WACV2024/papers/Choi_Dual_Domain_Diffusion_Guidance_for_3D_CBCT_Metal_Artifact_Reduction_WACV_2024_paper.pdf)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://openaccess.thecvf.com/content/WACV2024/papers/Choi_Dual_Domain_Diffusion_Guidance_for_3D_CBCT_Metal_Artifact_Reduction_WACV_2024_paper.pdf)

**Deep learning based projection domain metal segmentation for metal artifact reduction in cone beam computed tomography** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*H. Agrawal et al.* \
**Summary** \
The authors proposed to use noisy Monte Carlo simulations to train a U-Net for metal segmentation in the projection domain. Including both the crops and full size images in the training data improved the metal segmentation performance. The evaluations were conducted on real clinical CBCT data to show the reduction in metal artifacts after inpainting the segmented metal traces in the projections. The evaluations included challenging datasets with large high density objects, motion artifacts, and out of field of view metals. \
IEEE Access 2023. [[doi](https://ieeexplore.ieee.org/document/10250444)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://ieeexplore.ieee.org/document/10250444)

 **Quad-Net: Quad-domain Network for CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*Z. Li et al.* \
**Summary** \
In addition to the dual domain artifact reduction in sinogram and image domain, the paper proposed to also apply Fourier domain processing of features in sinogram and image domain, respectively.\
arXiv 2023 [[Paper](https://arxiv.org/abs/2207.11678)]. IEEE TMI 2024. [[doi](https://ieeexplore.ieee.org/document/10385220)]. [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/longzilicart/Quad-Net/tree/master)


**Unsupervised metal artifacts reduction network for CT images based on
efficient transformer** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*L. Zhu et al.*\
**Summary**
A method to incorporate efficient Transformer blocks in the two generators of the CycleGAN. Additionally, the method also includes a loss term for the forward projections in the non-metal area. The results are compared with linear interpolation, CycleGAN and ADN.\
BSPC. Nov, 2023. [[doi](https://doi.org/10.1016/j.bspc.2023.105753)]

**Simulation-driven training of vision transformers enables
metal artifact reduction of highly truncated CBCT scans** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*F. Fan et al.*  \
**Summary** \
The paper proposes to use Swin-transformer for metal segmentation in the projection domain for CBCT data. Simulation data is used for the training. U-Net is shown to work best when the test data is similiar to the train data. But when the test data source is different than training, Swin transformer performs better.\
Medical Physics 2023. [[doi](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16919)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16919)

**Metal Artifact Reduction In Cone-Beam Extremity Images Using Gated Convolutions** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*H. Agrawal et al.* \
**Summary** \
The authors propose to use Gated Convoltuions for the metal area inpainting in the projection domain. The method is compared against linear interpolation, U-Net, and Partial Convolutions. \
ISBI 2021. [[doi](https://ieeexplore.ieee.org/document/9434163)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://acris.aalto.fi/ws/portalfiles/portal/64983388/ELEC_Agrawal_etal_Metal_artifact_reduction_IEEE_ISBI2021_acceptedauthormanuscript.pdf)

**ADN: Artifact Disentanglement Network for
Unsupervised Metal Artifact Reduction** \
*H. Liao et al.* <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
**Summary** \
The authors propose to use a disentanglement network to separate the metal artifact and the non-metal artifact latents. Two encoders are used to conde the latents. The latents are used by the decoders to generate the metal artifact affected and metal artifact free images. The method is compared against linear interpolation, NMAR, U-Net, cGANMAR,and CycleGAN among other methods. U-Net had the best PSNR and cGANMAR had the best SSIM. ADN had best PSNR and SSIM among the compared unsupervised methods. \
arXiv 2019 [[Paper](https://arxiv.org/pdf/1908.01104.pdf)]. IEEE TMI 2019. [[doi](https://ieeexplore.ieee.org/document/8788607)] [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/liaohaofu/adn)