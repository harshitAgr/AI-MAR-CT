# AI-MAR-CT


 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

 Collection of deep learning-based metal artifact reduction (MAR) articles for CT/CBCT Imaging.

 1. **TDMAR-Net: A Frequency-Aware Tri-Domain Diffusion Network for CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*W. Chen, B. Ning, Z. Zhou, L. Shi, Q. Liu* \
**Summary** \
TDMAR-Net is an unsupervised diffusion model-based network that uniquely operates across three domains—projection (sinogram), image, and Fourier—to reduce metal artifacts in CT images. A high-pass filter module in the Fourier domain adjusts the weights of high-frequency and low-frequency components, while block-wise processing extracts diffusion priors that are iteratively introduced into the sinogram and image domains to fill metal-induced artifact regions. The framework employs a two-stage training strategy combining large-scale pretraining with masked data fine-tuning to enhance accuracy and adaptability. Validated on both synthetic and clinical datasets, TDMAR-Net demonstrates superior performance compared to existing unsupervised MAR methods. \
Physics in Medicine & Biology, 2025. [[doi](https://doi.org/10.1088/1361-6560/ae0efc)]

 1. **Bi-Constraints Diffusion: A Conditional Diffusion Model With Degradation Guidance for Metal Artifact Reduction** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*M. Luo, N. Zhou, T. Wang, L. He, W. Wang, H. Chen, P. Liao, Y. Zhang* \
**Summary** \
BCDMAR is an unsupervised metal artifact reduction method that combines iterative reconstruction with a conditional score-based diffusion model. Unlike traditional approaches that use metal-excluded projection operators, BCDMAR introduces a metal artifact degradation operator in the data-fidelity term and employs a pre-corrected image as a prior constraint to guide the diffusion model's generation process, effectively preventing grayscale shifts and unreliable structures. By iteratively applying the score-based diffusion model and the data-fidelity step in each sampling iteration, the method maintains reliable tissue representation around metal regions while producing highly consistent structures in non-metal regions. Extensive experiments demonstrate BCDMAR's superior performance over state-of-the-art unsupervised methods (ADN, ACDNet) and supervised methods (DuDoNet, InDuDoNet+) both quantitatively and visually. \
IEEE Transactions on Medical Imaging, 2025. [[doi](https://doi.org/10.1109/TMI.2024.3442950)]

1. **Unsupervised CT Metal Artifact Reduction by Plugging Diffusion Priors in Dual Domains** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*X. Liu, Y. Xie, S. Diao, S. Tan, X. Liang* \
**Summary** \
DuDoDp-MAR is an unsupervised dual-domain metal artifact reduction method that leverages diffusion model priors without requiring paired training data. The approach first trains a diffusion model on artifact-free CT images, then iteratively applies the learned priors in both sinogram and image domains to restore metal-corrupted regions. A key innovation is the use of temporally dynamic weight masks for image-domain fusion. Trained and evaluated on DeepLesion-based synthetic data (90 metal masks for training, 10 for testing) and clinical datasets, DuDoDp-MAR achieves the best overall performance among unsupervised methods, surpassing Score-MAR (another diffusion-based approach), ADN, and even the supervised CNNMAR, while demonstrating superior visual quality on clinical data. \
[[arXiv](https://arxiv.org/abs/2308.16742)] IEEE Transactions on Medical Imaging, 2024. [[doi](https://doi.org/10.1109/TMI.2024.3351201)] [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/DeepXuan/DuDoDp-MAR)

1. **DCDiff: Dual-Domain Conditional Diffusion for CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*R. Shen, X. Li, Y.-F. Li, C. Sui, Y. Peng, Q. Ke* \
**Summary** \
DCDiff is a supervised dual-domain diffusion framework for CT MAR that conditions the image generation process on both image-domain and sinogram-domain information. In the image domain, the raw metal-corrupted CT image and the filtered back-projection (FBP) of the metal trace are used as conditions; in the sinogram domain, a novel Diffusion Interpolation (DI) algorithm generates sinogram priors by training a dedicated diffusion model to inpaint the metal-corrupted regions. Two UNet-based denoising networks (~26M parameters total) are trained on paired metal-corrupted/metal-free CT images from the DeepLesion dataset. Ablation studies show that adding DI improves PSNR/SSIM from 31.59 dB/0.9124 to 35.47 dB/0.9390. Experimental results demonstrate that DCDiff outperforms LI, NMAR, DuDoNet, and InDuDoNet on both synthetic and clinical data. \
MICCAI, 2024. [[doi](https://doi.org/10.1007/978-3-031-72104-5_22)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://papers.miccai.org/miccai-2024/paper/1608_paper.pdf)

1. **UPGRADE-Net: Unsupervised Sinogram-domain Data-Consistent Network for Metal Artifact Reduction** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*Z. Wu et al.* \
**Summary** \
UPGRADE-Net tackles CT MAR directly in the sinogram by pairing a conditional denoising diffusion model (DDPM-MAR) with an unsupervised reverse pipeline that learns metal-trace statistics from pseudo masks injected into metal-free regions interpolated via LI. The diffusion model is regularized with two physics-driven constraints—conjugate-ray consistency to tie symmetric detector bins and accumulation-ray consistency to preserve per-view line integrals—so the network honors data fidelity without paired supervision. Trained on 40k synthetic cases assembled from DeepLesion slices (256×256, 361 views, 367 detectors) using 40 randomly sampled masks drawn from a library of 80 shapes (with the remaining 10 masks held out for 2k evaluation cases alongside real experimental scans), UPGRADE-Net lifts sinogram PSNR/SSIM to 33.44 dB / 0.961, surpassing FBP, LI, IMAR, SIDNN, Edge GAN, and matching supervised ACDNet while maintaining sharper lung and abdominal structures around large implants. \
IEEE Transactions on Medical Imaging, 2025. [[doi](https://doi.org/10.1109/TMI.2025.3630832)]

1. **UPMCL-Net: Unsupervised Projection-domain Multiview Constraint Learning for CBCT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*Z. Wu et al.* \
**Summary** \
UPMCL-Net directly learns to inpaint metal-corrupted CBCT projections without paired ground truth by combining a transformer-based MultiView Consistency Module (MVCM) that mines complementary anatomy across adjacent and interval views with a Hybrid Feature Attention Module (HFAM) that adaptively fuses intraview background cues and interview priors. Training is fully unsupervised: triangulation-based interpolation provides self-reconstruction targets, randomly generated pseudo masks expose the model to diverse missing patterns, and an adversarial loss sharpens fine textures in the synthesized metal traces. Trained on 25 of 30 intraoperative spine CBCT scans and evaluated on the remaining cases, UPMCL-Net surpasses LI, TRI, RegGAN, AOT-GAN, LBAM, and Palette on both simulated and clinical data. \
IEEE Transactions on Medical Imaging, 2025. [[doi](https://doi.org/10.1109/TMI.2025.3638630)]

1. **Dual-Domain Denoising Diffusion Probabilistic Model for Metal Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*W. Xia et al.* \
**Summary** \
This article proposes a supervised dual-domain Denoising Diffusion Probabilistic Model (DDPM) for CT MAR, operating in both image and sinogram domains. The initial input for the sinogram domain DDPM model is generated by first applying a linear interpolation to the metal-corrupted sinogram data, reconstructing it, refining it with a trained FBPConvNet, and forward-projecting it. After obtaining such initially corrected sinogram, the DDPM is trained. At each iteration of DDPM, the non-metal area is inserted into the model's output to refine only the metal area. After reconstruction, another DDPM is trained to refine the image domain. Instead of predicting the noise, the image domain DDPM predicts the residual between the input image and the metal-free image. To reduce high computational cost during the inference, denoising diffusion implicit model (DDIM) is used. The models are trained on the simulated data from CT-MAR AAPM Grand Challenge dataset. For the sinogram-domain DDPM, the sinogram was resized to 512x512 for processing and resized back to the original size of 1000x900 after processing. On the other hand, the image-domain DDPM operates on the original image size of 512x512 pixels. The results are compared with linear interpolation, NMAR, and DICDNet on simulated and one clinical dataset. On a Tesla V100 GPU, the proposed method takes about 29 seconds to process a single image, including the time for the sinogram and image domain DDIMs. \
IEEE Transactions on Radiation and Plasma Medical Sciences, 2025. [[doi](https://doi.org/10.1109/trpms.2025.3582528)]

1. **PRAISE-Net: Deep Projection-domain
Data-consistent Learning Network for CBCT Metal
Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*Z. Wu et al.* \
**Summary** \
PRAISE-Net is a deep learning framework for metal artifact reduction (MAR) in cone-beam computed tomography (CBCT), designed to ensure data consistency in the projection domain. It employs a Low2High strategy consisting of two stages. In the first stage, the Prior Information-Guided Denoising Diffusion Probabilistic Model (PIG-DDPM) performs low-resolution inpainting of metal-corrupted regions. The inputs to this model are the projection data affected by metal artifacts and a prior estimate generated using linear interpolation; the target is the corresponding metal-free projection data. In the second stage, a Super-Resolution Reconstruction (SRR) module based on a Swin Transformer takes the inpainted low-resolution projections and reconstructs high-resolution projections. The target for this stage is the high-resolution metal-free projection data. To improve generalization to real clinical CBCT data, PRAISE-Net incorporates a CBCT Domain Adaptation (CBCT-DA) module into the training phase of PIG-DDPM. CBCT-DA uses two encoders to disentangle domain-invariant anatomical features from domain-specific features. It employs adversarial loss through a domain discriminator to align the feature distributions of simulated and clinical data, and uses mutual information loss to enforce separation between anatomical and domain-relevant representations. Importantly, CBCT-DA is used only during training to enable domain-invariant feature learning; it is not required during testing, where the trained model can be directly applied to clinical CBCT data. After high-resolution projections are restored, the final CBCT images are reconstructed using the Feldkamp-Davis-Kress (FDK) algorithm. \
IEEE Transactions on Instrumentation and Measurement, 2025. [[doi](https://doi.org/10.1109/TIM.2025.3551446)]

1. **Radiologist-in-the-Loop Self-Training for Generalizable CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Self-Supervised-blue.svg" alt="Self-Supervised"> \
*C. Ma et al.* \
**Summary** \
RISE-MAR is a novel radiologist-in-the-loop self-training framework for generalizable CT metal artifact reduction (MAR), addressing the domain gap and confirmation bias issues in existing supervised and semi-supervised MAR methods. The framework integrates a Clinical Quality Assessor (CQA)—a transformer-based network trained on radiologist-annotated CT images—that evaluates MAR outputs and ensures only high-quality pseudo ground-truths are used for training. CQA employs a spatial-frequency token mixer combining convolutional and self-attention mechanisms for multi-scale feature extraction, followed by a vectorization layer and quality head that predicts clinical quality scores. The self-training framework utilizes a teacher-student architecture, where the teacher network, pretrained on simulated MAR data, generates pseudo ground-truths for real clinical images. These pseudo ground-truths are assessed by CQA, and only high-quality ones (within a defined threshold) train the student MAR network via a supervised loss (L_sim) on paired simulated data and an unsupervised loss on real clinical images. The teacher model is progressively updated using an Exponential Moving Average (EMA) of the student’s improved weights, ensuring continuous refinement of pseudo ground-truths. Extensive experiments on multiple clinical datasets demonstrate RISE-MAR's superior generalization over state-of-the-art methods in out-of-domain datasets.\
[[arXiv](https://arxiv.org/pdf/2501.15610)]. IEEE TMI, 2025. [[doi](https://ieeexplore.ieee.org/abstract/document/10857416)]

1. **Implicit neural representation-based method for metal-induced beam hardening artifact reduction in X-rayCT imaging** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*H. S. Park et al.* \
**Summary** \
This paper introduces a parameter-free metal-induced beam hardening correction (MBHC) method for CT imaging utilizing implicit neural representations (INRs). In contrast to the MBHC approach, the proposed method generates two tomographic images: one representing the monochromatic attenuation distribution and the other capturing nonlinear beam hardening effects. This eliminates the need for explicit metal segmentation and parameter estimation. The INR-generated images are combined to simulate forward projection data, which is then used to compute a loss function for network training. Experimental results demonstrate effective reduction of beam hardening artifacts arising from interactions between metals, bone, and teeth. Furthermore, the method exhibits potential for addressing challenges associated with photon starvation and truncated fields-of-view, leading to improved image reconstruction quality. \
Medical Physics, 2025. [[doi](doi/10.1002/mp.17649)]

1. **Metal implant segmentation in CT images based on diffusion model** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*K. Xie et al.* \
**Summary** \
The authors propose to use a diffusion model "DiffSeg" to segment the metal implants in CT images. The method is compared against U-Net, Attention U-Net, R2U-Net, and DeepLabV3+. The training and testing is conducted using simulated data. Additionally, the method is also tested on real clinical data. The method is shown to outperform the other methods. Better segmentation of metals is shown to reduce the artifacts when using NMAR. \
BMC Medical Imaging, 2024. [[doi]( https://doi.org/10.1186/s12880-024-01379-1)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-024-01379-1) 
 
1. **A denoising diffusion probabilistic model for metal artifact reduction in CT**  <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*G. M. Karageorgos et al.* \
**Summary** \
The authors propose to use a denoising diffusion probabilistic model (DDPM) for metal artifact reduction in CT. The model was used to inpaint the metal traces in the projection domain. The training was done in the unsupervised manner but during the inference, the segmented metal trace was required. The DDPM model was compared with a Partial Convolutions-based U-Net and a Gated Convolutions-based GAN. The proposed method showed promising results, however, the method was computationally expensive. Moreover, it was less effective in reducing metal artifacts caused by large metal objects. The authors also noted that it is crucial to have an accurate metal trace segmentation for the best performance of the proposed network\
IEEE TMI, 2024. [[doi]](https://ieeexplore.ieee.org/document/10586949)

1. **DiffMAR: A Generalized Diffusion Model for Metal Artifact Reduction in CT images**  <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*T. Cai et al.* \
**Summary** \
DiffMAR frames CT metal artifact reduction as reversing a physically motivated *linear degradation* process that mimics metal-artifact formation, and learns an iterative restoration path with a conditional diffusion model trained on paired metal-corrupted/metal-free CT images. To reduce error accumulation across the multi-step reverse process, it introduces a Time-Latent Adjustment (TLA) module that adaptively adjusts time embeddings at the latent level. To better preserve anatomy (and avoid over-smoothing), it adds a Structure Information Extraction (SIE) module that extracts structural priors from the linearly-interpolated (LI) correction in the image domain and injects them at each restoration step. On a DeepLesion-based synthetic benchmark (1,000 training images; 90 training masks; 2,000 test images; images resized to 416×416 with 640 views) and clinical testing (SpineWeb plus an in-house dental CT set), DiffMAR outperforms LI, DuDoNet, InDuDoNet(+), CycleGAN/Pix2PixGAN, and DICDNet/ACDNet, improving average PSNR/SSIM from 45.31 dB / 0.9899 (ACDNet) to 46.28 dB / 0.9922 and lowering RMSE from 0.0133 to 0.0119 on the synthetic test set. \
IEEE Journal of Biomedical and Health Informatics, 2024. [[doi](https://doi.org/10.1109/JBHI.2024.3439729)]

1. **PND-Net: Physics-inspired Non-local Dual-domain Network for Metal Artifact Reduction**  <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
 *J. Xia et al.* \
 **Summary**\
 The authors propose to use three networks. One is called non-local sinogram decomposition network (NSD-Net) to acquire the weighted artifact component in sinogram domain. The second network is an image restoration network (IR-Net) to reduce the residual and secondary artifacts in the image domain. The third trainable fusion network (F-Net) in the artifact synthesis path is used for unpaired learning. The method is compared with linear interpolation, NMAR, DuDoNet, DuDoNet++, InDuDoNet+, ADN, and U-DuDoNet.\
 IEEE TMI, 2024. [[doi](https://ieeexplore.ieee.org/document/10404006)] [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/Ballbo5354/PND-Net/tree/main)


1. **Dual Domain Diffusion Guidance for 3D CBCT Metal Artifact Reduction**  <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*Y. Choi et al.* \
**Summary** \
The authors propose to use two 2D diffusion models in image domain to synthesize metal artifacts and metal free images, respectively. The combination of metal artifact and metal free images is forward projected and a guidance from the error in the projection domain is used. The results are compared with InDuDoNet+, ACDNet, FEL, and Blind DPS methods.  \
WACV, 2024. [[doi](https://openaccess.thecvf.com/content/WACV2024/papers/Choi_Dual_Domain_Diffusion_Guidance_for_3D_CBCT_Metal_Artifact_Reduction_WACV_2024_paper.pdf)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://openaccess.thecvf.com/content/WACV2024/papers/Choi_Dual_Domain_Diffusion_Guidance_for_3D_CBCT_Metal_Artifact_Reduction_WACV_2024_paper.pdf)

1. **Polyner: Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*Q. Wu et al.* \
**Summary** \
Polyner treats MAR as a nonlinear reconstruction problem, embedding a differentiable polychromatic CT forward model and an energy-dependent smoothness prior inside an implicit neural representation (hash-encoded MLP) so each scan can be optimized end-to-end without external training data. During inference, 80 rays are randomly sampled per iteration and the network learns case-specific attenuation fields across up to 101 energy bins in ~2 minutes for a 256×256 slice, preserving metal-trace signals instead of masking them. On synthetic DeepLesion data it reaches 37.57 dB / 0.975 SSIM (2nd overall) and on the OOD XCOM benchmark it tops all baselines at 38.74 dB / 0.951 SSIM, outperforming supervised CNN-MAR/DICDNet/ACDNet. The same per-case optimization transfers to real micro-CT walnuts with clips and mouse tibia scans, removing residual streaks left by FBP, LI, Score-MAR, and even ACDNet. \
NeurIPS, 2023. [[doi](https://dl.acm.org/doi/abs/10.5555/3666122.3669170)][![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/iwuqing/Polyner)[![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://proceedings.neurips.cc/paper_files/paper/2023/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf)

1. **Deep learning based projection domain metal segmentation for metal artifact reduction in cone beam computed tomography** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*H. Agrawal et al.* \
**Summary** \
The authors proposed to use noisy Monte Carlo simulations to train a U-Net for metal segmentation in the projection domain. Including both the crops and full size images in the training data improved the metal segmentation performance. The evaluations were conducted on real clinical CBCT data to show the reduction in metal artifacts after inpainting the segmented metal traces in the projections. The evaluations included challenging datasets with large high density objects, motion artifacts, and out of field of view metals. \
IEEE Access, 2023. [[doi](https://ieeexplore.ieee.org/document/10250444)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://ieeexplore.ieee.org/document/10250444)

1. **Quad-Net: Quad-domain Network for CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*Z. Li et al.* \
**Summary** \
In addition to dual-domain artifact reduction in the sinogram and image domains, the paper also proposes applying Fourier domain processing to features in both domains.\
[[arXiv](https://arxiv.org/abs/2207.11678)]. IEEE TMI, 2024. [[doi](https://ieeexplore.ieee.org/document/10385220)]. [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/longzilicart/Quad-Net/tree/master)


1. **Unsupervised metal artifacts reduction network for CT images based on efficient transformer** <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
*L. Zhu et al.*\
**Summary**
A method to incorporate efficient Transformer blocks in the two generators of the CycleGAN. Additionally, the method also includes a loss term for the forward projections in the non-metal area. The results are compared with linear interpolation, CycleGAN and ADN.\
BSPC, 2023. [[doi](https://doi.org/10.1016/j.bspc.2023.105753)]

1. **Simulation-driven training of vision transformers enables metal artifact reduction of highly truncated CBCT scans** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*F. Fan et al.*  \
**Summary** \
The paper proposes to use Swin-transformer for metal segmentation in the projection domain for CBCT data. Simulation data is used for the training. U-Net is shown to work best when the test data is similiar to the train data. But when the test data source is different than training, Swin transformer performs better.\
Medical Physics, 2023. [[doi](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16919)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16919)

1. **DL-based inpainting for metal artifact reduction for cone beam CT using metal path length information** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*T. M. Gottschalk et al.* \
**Summary** \
The authors propose to use a U-Net for inpainting the metal traces in the projection domain. The metal path length information is used to guide the inpainting along with the metal mask. Authors suggest that it is beneficial to not discard the information inside the metal mask. Additionally, 10 nearby projections were also concatenated to the input, making the total number of channels in the input 13. The output was the projection to be inpainted/corrected.\
Medical Physics, 2022. [[doi](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15909)]

1. **A fidelity-embedded learning for metal artifact reduction in dental CBCT** <img src="https://img.shields.io/badge/Iterative-blue.svg" alt="Iterative"> \
*H. S. Park et al.* \
**Summary** \
The authors propose using an iterative method for fidelity-embedded learning to reduce metal artifacts in dental CBCT images. In each iteration, the network learns to predict the remaining metal artifact within the image. Additionally, a data-fidelity error term is calculated in the projection domain, but only for the non-metal areas. The Astra Toolbox was utilized for implementation. Training was conducted on simulated data, whereas testing was performed on both simulated and real clinical data. \
Medical Physics, 2022. [[doi](https://doi.org/10.1002/mp.15720)] 


1. **Metal Artifact Reduction In Cone-Beam Extremity Images Using Gated Convolutions** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*H. Agrawal et al.* \
**Summary** \
The authors propose to use Gated Convolutions for the metal area inpainting in the projection domain. The method is compared against linear interpolation, U-Net, and Partial Convolutions. \
ISBI, 2021. [[doi](https://ieeexplore.ieee.org/document/9434163)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://acris.aalto.fi/ws/portalfiles/portal/64983388/ELEC_Agrawal_etal_Metal_artifact_reduction_IEEE_ISBI2021_acceptedauthormanuscript.pdf)

1. **Multiple Window Learning for CT Metal Artifact Reduction** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
*C. Niu et al.* \
**Summary** \
The authours proposed a deep learning-based framework to enhance CT MAR by leveraging information across multiple Hounsfield Unit (HU) windows. They argued that traditional approaches typically normalize CT data into a single HU window, which can compromise training effectiveness due to unequal emphasis on different HU ranges—an important consideration in clinical applications where tissues are best visualized under specific window settings.  Multiple Window Learning Network (MWLNet) addresses this by employing multiple convolutional neural network (CNN) branches, each tailored to a specific HU window. Inter-branch communication is facilitated by Window Transfer (WT) layers, which normalize and clip image data to map across HU ranges, enabling end-to-end training with gradients flowing between branches to enforce learning across scales. The training strategy uses an L1 loss summed across all window branches to ensure balanced learning, with three specific windows selected: [-1000, 2000], [-320, 480], and [-160, 240], though the method is flexible to other configurations. Training utilized paired artifact-free and artifact-affected images generated via CatSim-based simulation. MWLNet was evaluated on simulated CT scans of spine, teeth, and hip regions, as well as clinical datasets, with all images being 512×512 slices exhibiting varying artifact levels. Results showed that MWLNet outperformed NMAR and single-window CNN baselines (SWLNets) in PSNR, SSIM, and visual quality, notably preserving fine structures in narrow windows while reducing artifacts more effectively. Intermediate results validated that contextual information from larger HU windows improves prediction accuracy in smaller windows.
Proc. SPIE, 2021. [[doi]( https://doi.org/10.1117/12.2596239)]



1. **ADN: Artifact Disentanglement Network for
Unsupervised Metal Artifact Reduction** \
*H. Liao et al.* <img src="https://img.shields.io/badge/Unsupervised-blue.svg" alt="Unsupervised"> \
**Summary** \
The authors propose to use a disentanglement network to separate the metal artifact and the non-metal artifact latent variables. Two encoders are used to encode the latent variables. The latent variables are used by the decoders to generate the metal artifact affected and metal artifact free images. The method is compared against linear interpolation, NMAR, U-Net, cGANMAR, and CycleGAN among other methods. U-Net had the best PSNR and cGANMAR had the best SSIM. ADN had best PSNR and SSIM among the compared unsupervised methods. \
[[arxiv](https://arxiv.org/pdf/1908.01104.pdf)]. IEEE TMI, 2019. [[doi](https://ieeexplore.ieee.org/document/8788607)] [![Code](https://img.shields.io/badge/Code-purple.svg)](https://github.com/liaohaofu/adn)
