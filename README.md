# AI-MAR-CT


 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

 Collection of deep learning-based metal artifact reduction (MAR) articles for CT/CBCT Imaging.

 **PRAISE-Net: Deep Projection-domain
Data-consistent Learning Network for CBCT Metal
Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*Z. Wu et al.* \
**Summary** \
PRAISE-Net is a deep learning framework for metal artifact reduction (MAR) in cone-beam computed tomography (CBCT), designed to ensure data consistency in the projection domain. It employs a Low2High strategy consisting of two stages. In the first stage, the Prior Information-Guided Denoising Diffusion Probabilistic Model (PIG-DDPM) performs low-resolution inpainting of metal-corrupted regions. The inputs to this model are the projection data affected by metal artifacts and a prior estimate generated using linear interpolation; the target is the corresponding metal-free projection data. In the second stage, a Super-Resolution Reconstruction (SRR) module based on a Swin Transformer takes the inpainted low-resolution projections and reconstructs high-resolution projections. The target for this stage is the high-resolution metal-free projection data. To improve generalization to real clinical CBCT data, PRAISE-Net incorporates a CBCT Domain Adaptation (CBCT-DA) module into the training phase of PIG-DDPM. CBCT-DA uses two encoders to disentangle domain-invariant anatomical features from domain-specific features. It employs adversarial loss through a domain discriminator to align the feature distributions of simulated and clinical data, and uses mutual information loss to enforce separation between anatomical and domain-relevant representations. Importantly, CBCT-DA is used only during training to enable domain-invariant feature learning; it is not required during testing, where the trained model can be directly applied to clinical CBCT data. After high-resolution projections are restored, the final CBCT images are reconstructed using the Feldkamp-Davis-Kress (FDK) algorithm. \
IEEE Transactions on Instrumentation and Measurement, 2025. [[doi](https://doi.org/10.1109/TIM.2025.3551446)]

**Radiologist-in-the-Loop Self-Training for Generalizable CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Self-Supervised-blue.svg" alt="Self-Supervised"> \
*C. Ma et al.* \
**Summary** \
RISE-MAR is a novel radiologist-in-the-loop self-training framework for generalizable CT metal artifact reduction (MAR), addressing the domain gap and confirmation bias issues in existing supervised and semi-supervised MAR methods. The framework integrates a Clinical Quality Assessor (CQA)—a transformer-based network trained on radiologist-annotated CT images—that evaluates MAR outputs and ensures only high-quality pseudo ground-truths are used for training. CQA employs a spatial-frequency token mixer combining convolutional and self-attention mechanisms for multi-scale feature extraction, followed by a vectorization layer and quality head that predicts clinical quality scores. The self-training framework utilizes a teacher-student architecture, where the teacher network, pretrained on simulated MAR data, generates pseudo ground-truths for real clinical images. These pseudo ground-truths are assessed by CQA, and only high-quality ones (within a defined threshold) train the student MAR network via a supervised loss (L_sim) on paired simulated data and an unsupervised loss on real clinical images. The teacher model is progressively updated using an Exponential Moving Average (EMA) of the student’s improved weights, ensuring continuous refinement of pseudo ground-truths. Extensive experiments on multiple clinical datasets demonstrate RISE-MAR's superior generalization over state-of-the-art methods in out-of-domain datasets.\
[[arXiv](https://arxiv.org/pdf/2501.15610)]. IEEE TMI, 2025. [[doi](https://ieeexplore.ieee.org/abstract/document/10857416)]

 **Implicit neural representation-based method formetal-induced beam hardening artifact reduction in X-rayCT imaging** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*H. S. Park et al.* \
**Summary** \
This paper introduces a parameter-free metal-induced beam hardening correction (MBHC) method for CT imaging utilizing implicit neural representations (INRs). In contrast to the MBHC approach, the proposed method generates two tomographic images: one representing the monochromatic attenuation distribution and the other capturing nonlinear beam hardening effects. This eliminates the need for explicit metal segmentation and parameter estimation. The INR-generated images are combined to simulate forward projection data, which is then used to compute a loss function for network training. Experimental results demonstrate effective reduction of beam hardening artifacts arising from interactions between metals, bone, and teeth. Furthermore, the method exhibits potential for addressing challenges associated with photon starvation and truncated fields-of-view, leading to improved image reconstruction quality. \
Medical Physics, 2025. [[doi](doi/10.1002/mp.17649)]

**Metal implant segmentation in CT images based on diffusion model** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*K. Xie et al.* \
**Summary** \
The authors propose to use a diffusion model "DiffSeg" to segment the metal implants in CT images. The method is compared against U-Net, Attention U-Net, R2U-Net, and DeepLabV3+. The training and testing is conducted using simulated data. Additionally, the method is also tested on real clinical data. The method is shown to outperform the other methods. Better segmentation of metals is shown to reduce the artifacts when using NMAR. \
BMC Medical Imaging, 2024. [[doi]( https://doi.org/10.1186/s12880-024-01379-1)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-024-01379-1) 
 
 **A denoising diffusion probabilistic model for metal artifact reduction in CT**  <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*G. M. Karageorgos et al.* \
**Summary** \
The authors propose to use a denoising diffusion probabilistic model (DDPM) for metal artifact reduction in CT. The model was used to inpaint the metal traces in the projection domain. The training was done in the unsupervised manner but during the inference, the segmented metal trace was required. The DDPM model was compared with a Partial Convolutions-based U-Net and a Gated Convolutions-based GAN. The proposed method showed promising results, however, the method was computationally expensive. Moreover, it was less effective in reducing met artifacts caused by large metal objects. The authors also noted that it is crucial to have an accurate metal trace segmentation for the best performance of the proposed network\
IEEE TMI, 2024. [[doi]](https://ieeexplore.ieee.org/document/10586949)

 **PND-Net: Physics-inspired Non-local Dual-domain Network for Metal Artifact Reduction**  <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
 *J. Xia et al.* \
 **Summary**\
 The authors propose to use three networks. One is called non-local sinogram decomposition network (NSD-Net) to acquire the weighted artifact component in sinogram domain. The second network is an image restoration network (IR-Net) to reduce the residual and secondary artifacts in the image domain. The third trainable fusion network (F-Net) in the artifact synthesis path is used for unpaired learning. The method is compared with linear interpolation, NMAR, DuDoNet, DuDoNet++, InDuDoNet+, ADN, and U-DuDoNet.\
 IEEE TMI, 2024. [[doi](https://ieeexplore.ieee.org/document/10404006)] [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/Ballbo5354/PND-Net/tree/main)


**Dual Domain Diffusion Guidance for 3D CBCT Metal Artifact Reduction**  <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*Y. Choi et al.* \
**Summary** \
The authors propose to use two 2D diffusion models in image domain to synthesize metal artifacts and metal free images, respectively. The combination of metal artifact and metal free images is forward projected and a guidance from the error in the projection domain is used. The results are compared with InDuDoNet+, ACDNet, FEL, and Blind DPS methods.  \
WACV, 2024. [[doi](https://openaccess.thecvf.com/content/WACV2024/papers/Choi_Dual_Domain_Diffusion_Guidance_for_3D_CBCT_Metal_Artifact_Reduction_WACV_2024_paper.pdf)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://openaccess.thecvf.com/content/WACV2024/papers/Choi_Dual_Domain_Diffusion_Guidance_for_3D_CBCT_Metal_Artifact_Reduction_WACV_2024_paper.pdf)

**Deep learning based projection domain metal segmentation for metal artifact reduction in cone beam computed tomography** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*H. Agrawal et al.* \
**Summary** \
The authors proposed to use noisy Monte Carlo simulations to train a U-Net for metal segmentation in the projection domain. Including both the crops and full size images in the training data improved the metal segmentation performance. The evaluations were conducted on real clinical CBCT data to show the reduction in metal artifacts after inpainting the segmented metal traces in the projections. The evaluations included challenging datasets with large high density objects, motion artifacts, and out of field of view metals. \
IEEE Access, 2023. [[doi](https://ieeexplore.ieee.org/document/10250444)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://ieeexplore.ieee.org/document/10250444)

 **Quad-Net: Quad-domain Network for CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*Z. Li et al.* \
**Summary** \
In addition to the dual domain artifact reduction in sinogram and image domain, the paper proposed to also apply Fourier domain processing of features in sinogram and image domain, respectively.\
[[arXiv](https://arxiv.org/abs/2207.11678)]. IEEE TMI, 2024. [[doi](https://ieeexplore.ieee.org/document/10385220)]. [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/longzilicart/Quad-Net/tree/master)


**Unsupervised metal artifacts reduction network for CT images based on
efficient transformer** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*L. Zhu et al.*\
**Summary**
A method to incorporate efficient Transformer blocks in the two generators of the CycleGAN. Additionally, the method also includes a loss term for the forward projections in the non-metal area. The results are compared with linear interpolation, CycleGAN and ADN.\
BSPC, 2023. [[doi](https://doi.org/10.1016/j.bspc.2023.105753)]

**Simulation-driven training of vision transformers enables
metal artifact reduction of highly truncated CBCT scans** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*F. Fan et al.*  \
**Summary** \
The paper proposes to use Swin-transformer for metal segmentation in the projection domain for CBCT data. Simulation data is used for the training. U-Net is shown to work best when the test data is similiar to the train data. But when the test data source is different than training, Swin transformer performs better.\
Medical Physics, 2023. [[doi](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16919)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16919)

**DL-based inpainting for metal artifact reduction for cone beam CT using metal path length information** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*T. M. Gottschalk et al.* \
**Summary** \
The authors propose to use a U-Net for inpainting the metal traces in the projection domain. The metal path length information is used to guide the inpainting along with the metal mask. Authors suggest that it is benificial to not dicard the information inside the metal mask. Additionally, 10 nearby projections were also concatenated to the input, making the total number of channels in the input 13. The output was the projection to be inpainted/corrected.\
Medical Physics, 2022. [[doi](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15909)]

**A fidelity-embedded learning for metal artifact reduction in dental CBCT** <img src="https://img.shields.io/badge/Iterative-blue.svg" alt="Iterative"> \
*H. S. Park et al.* \
**Summary** \
The authors propose using an iterative method for fidelity-embedded learning to reduce metal artifacts in dental CBCT images. In each iteration, the network learns to predict the remaining metal artifact within the image. Additionally, a data-fidelity error term is calculated in the projection domain, but only for the non-metal areas. The Astra Toolbox was utilized for implementation. Training was conducted on simulated data, whereas testing was performed on both simulated and real clinical data. \
Medical Phyics, 2022. [[doi](https://doi.org/10.1002/mp.15720)] 


**Metal Artifact Reduction In Cone-Beam Extremity Images Using Gated Convolutions** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*H. Agrawal et al.* \
**Summary** \
The authors propose to use Gated Convoltuions for the metal area inpainting in the projection domain. The method is compared against linear interpolation, U-Net, and Partial Convolutions. \
ISBI, 2021. [[doi](https://ieeexplore.ieee.org/document/9434163)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://acris.aalto.fi/ws/portalfiles/portal/64983388/ELEC_Agrawal_etal_Metal_artifact_reduction_IEEE_ISBI2021_acceptedauthormanuscript.pdf)

**Multiple Window Learning for CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*C. Niu et al.* \
**Summary** \
The authours proposed a deep learning-based framework to enhance CT MAR by leveraging information across multiple Hounsfield Unit (HU) windows. They argued that traditional approaches typically normalize CT data into a single HU window, which can compromise training effectiveness due to unequal emphasis on different HU ranges—an important consideration in clinical applications where tissues are best visualized under specific window settings.  Multiple Window Learning Network (MWLNet) addresses this by employing multiple convolutional neural network (CNN) branches, each tailored to a specific HU window. Inter-branch communication is facilitated by Window Transfer (WT) layers, which normalize and clip image data to map across HU ranges, enabling end-to-end training with gradients flowing between branches to enforce learning across scales. The training strategy uses an L1 loss summed across all window branches to ensure balanced learning, with three specific windows selected: [-1000, 2000], [-320, 480], and [-160, 240], though the method is flexible to other configurations. Training utilized paired artifact-free and artifact-affected images generated via CatSim-based simulation. MWLNet was evaluated on simulated CT scans of spine, teeth, and hip regions, as well as clinical datasets, with all images being 512×512 slices exhibiting varying artifact levels. Results showed that MWLNet outperformed NMAR and single-window CNN baselines (SWLNets) in PSNR, SSIM, and visual quality, notably preserving fine structures in narrow windows while reducing artifacts more effectively. Intermediate results validated that contextual information from larger HU windows improves prediction accuracy in smaller windows.
Proc. SPIE, 2021. [[doi]( https://doi.org/10.1117/12.2596239)]



1\. **ADN: Artifact Disentanglement Network for
Unsupervised Metal Artifact Reduction** \
*H. Liao et al.* <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
**Summary** \
The authors propose to use a disentanglement network to separate the metal artifact and the non-metal artifact latents. Two encoders are used to conde the latents. The latents are used by the decoders to generate the metal artifact affected and metal artifact free images. The method is compared against linear interpolation, NMAR, U-Net, cGANMAR,and CycleGAN among other methods. U-Net had the best PSNR and cGANMAR had the best SSIM. ADN had best PSNR and SSIM among the compared unsupervised methods. \
[[arxiv](https://arxiv.org/pdf/1908.01104.pdf)]. IEEE TMI, 2019. [[doi](https://ieeexplore.ieee.org/document/8788607)] [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/liaohaofu/adn)
