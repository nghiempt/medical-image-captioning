# Medical Image Captioning

## Introduction

Medical Image Captioning (MIC) is a challenging task in the realm of artificial intelligence, aimed at automatically generating descriptive and accurate textual descriptions for medical images. This process involves several mandatory techniques to ensure robust performance and meaningful outputs.


## Software implementation

> Briefly describe the software that was written to produce the results of this
> paper.

All source code used to generate the results and figures in the paper are in
the `code` folder.
The calculations and figure generation are all run inside
[Jupyter notebooks](http://jupyter.org/).
The data used in this study is provided in `data` and the sources for the
manuscript text and figures are in `manuscript`.
Results generated by the code are saved in `results`.
See the `README.md` files in each directory for a full description.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/nghiempt/medical-image-captioning

or [download a zip archive](https://github.com/nghiempt/medical-image-captioning/archive/refs/heads/main.zip).

A copy of the repository is also archived at *insert DOI here*


## Approach

. . .

## Evaluation

1. **BLEU**

   > Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). Bleu: a method for automatic evaluation of machine translation. In *Proceedings of meeting of the association for computational linguistics* (pp. 311–318). 

   BLEU has frequently been reported as correlating well with human judgement, and remains a benchmark for the assessment of any new evaluation metric. There are however a number of criticisms that have been voiced. It has been noted that, although in principle capable of evaluating translations of any language, BLEU cannot, in its present form, deal with languages lacking word boundaries. It has been argued that although BLEU has significant advantages, there is no guarantee that an increase in BLEU score is an indicator of improved translation quality.

2. **ROUGE-L**

   > Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *Proceedings of meeting of the association for computational linguistics* (pp. 74–81). 

   ROUGE-L: Longest Common Subsequence (LCS) based statistics. [Longest common subsequence problem](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem) takes into account sentence level structure similarity naturally and identifies longest co-occurring in sequence n-grams automatically.


## Dataset detail
| id            | image | caption |
| ------------- | -------------  | ------------ |
| 1             | 20586908_6c613a14b80a8591_MG_R_CC_ANON.jpg | một nốt nằm ở vú trái có đường kính 25 mm, với... |
| 2             | 20586934_6c613a14b80a8591_MG_L_CC_ANON.jpg | một nốt nằm ở vú trái có đường kính 25 mm, với... |
| 3             | 20586960_6c613a14b80a8591_MG_R_ML_ANON.jpg | một nốt nằm ở vú trái có đường kính 25 mm, với... |
| 4             | 20586986_6c613a14b80a8591_MG_L_ML_ANON.jpg | một nốt nằm ở vú trái có đường kính 25 mm, với... |
| 5             | 20587054_b6a4f750c6df4f90_MG_R_CC_ANON.jpg | Nhũ ảnh ghi lại một cụm vi vôi hóa nằm ở vú ph... |
| 6             | 20587080_b6a4f750c6df4f90_MG_R_ML_ANON.jpg | Nhũ ảnh ghi lại một cụm vi vôi hóa nằm ở vú ph... |
| 7             | 20587148_fd746d25eb40b3dc_MG_R_CC_ANON.jpg | Nhũ ảnh ghi lại một cụm vi vôi hóa nằm ở vú ph... |
| 8             | 20587174_fd746d25eb40b3dc_MG_L_CC_ANON.jpg | không có hình ảnh nốt nào gợi ý ác tính, vôi ... |
| 9             | 20587200_fd746d25eb40b3dc_MG_R_ML_ANON.jpg | không có hình ảnh nốt nào gợi ý ác tính, vôi ... |
| 10            | 20587226_fd746d25eb40b3dc_MG_L_ML_ANON.jpg | không có hình ảnh nốt nào gợi ý ác tính, vôi ... |

Full dataset can be found in [kaggle](https://www.kaggle.com/datasets/nghiemthanhpham/inbreast).
