# DSformer

This github repository corresponds to our paper published in CIKM 2023(Dsformer: A double sampling transformer for multivariate time series long-term prediction).

To unified manage all baselines and models from our lab, DSformer's code will be stored together with other baselines at the following link: https://github.com/zezhishao/BasicTS

In the near future, I will upload the code (with commercial approval) to the above link. If you are interested in the DSformer, please get it from the corresponding link. 

The core hyperparameters include the following parts:

Input_len: History length

out_len：Future length

num_id：Number of variables

num_layer：Number of layers

muti_head：Number of muti_head attention

dropout：dropout

num_samp：Number of subsequence

IF_node: Whether to use variable embedding


If the code is helpful to you, please cite the following paper:
```bibtex
@inproceedings{yu2023dsformer,
  title={Dsformer: A double sampling transformer for multivariate time series long-term prediction},
  author={Yu, Chengqing and Wang, Fei and Shao, Zezhi and Sun, Tao and Wu, Lin and Xu, Yongjun},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={3062--3072},
  year={2023}
}
```
