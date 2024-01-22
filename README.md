# AI-MAR-CT


 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

 Collection of deep learning-based metal artifact reduction articles for CT/CBCT Imaging


 **Quad-Net: Quad-domain Network for CT Metal Artifact Reduction** \
*Z. Li et al.* \
**Summary** \
In addition to the dual domain artifact reduction in sinogram and image domain, the paper proposed to also apply Fourier domain processing of features in sinogram and image domain, respectively.\
arXiv 2023. [[Paper](https://arxiv.org/abs/2207.11678)] 
[[code](https://github.com/longzilicart/Quad-Net/tree/master)] \
IEEE TMI 2024. [[doi](https://ieeexplore.ieee.org/document/10385220)]

**Unsupervised metal artifacts reduction network for CT images based on
efficient transformer** \
L. Zhu et al.\
**Summary**
A memthod to incorporate efficient Transformer blocks in the two generators of the CycleGAN. Additionally, the method also includes a loss term for the forward projections in the non-metal area. The results are compared with linear interpolation, CycleGAN and ADN.\
BSPC. Nov, 2023. [[doi](https://doi.org/10.1016/j.bspc.2023.105753)]

**Simulation-driven training of vision transformers enables
metal artifact reduction of highly truncated CBCT scans** \
F. Fan et al.\
**Summary** \
The paper proposes to use Swin-transformer for metal segmentation in the projection domain for CBCT data. Simulation data is used for the training. U-Net is shown to work best when the test data is similiar to the train data. But when the test data source is different than training, Swin transformer performs better.\
arxiv 2022. [[Paper](https://arxiv.org/abs/2203.09207)] \
Medical Physics 2023. [[doi](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16919)] [![Open Access](https://img.shields.io/badge/Open%20Access-brightgreen.svg)](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16919)