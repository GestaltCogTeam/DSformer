# DSformer

This github repository corresponds to our paper published in CIKM 2023(Dsformer: A double sampling transformer for multivariate time series long-term prediction).

To unified manage all baselines and models from our lab, DSformer's code will be stored together with other baselines at the following link: https://github.com/zezhishao/BasicTS

The complete parameter Settings and training pipline are stored in the above link.

The current repository stores the model files for DSformer. Please note that we have optimized the code for DSformer in order to comply with commercial regulations. After testing, the current version has improved performance.

The core hyperparameters include the following parts:
- Input_len: History length
- out_len：Future length
- num_id：Number of variables
- num_layer：Number of layers. 1 or 2   (Note: In most cases, 1 is enough)
- muti_head：Number of muti_head attention. 1 to 4  (Note: In most cases, 1 or 2 is enough)
- dropout：dropout. 0.15 to 0.3
- num_samp：Number of subsequence. 2 or 3
- IF_node: Whether to use variable embedding. True or False (Note: In most cases, set to True)

In addition, some hyperparameters related to the learning rate are given as follows:
- Initial learning rate: 0.0002 (Note: In the team's BasicTS environment, setting to 0.002 might be better. In this case, milestone = [1,5,15,25,50,75,100], gamme = 0.5)
- Learning rate decay strategy：MultiStepLR
- milestone = [1,15,25,50,75,100], gamme = 0.5
- clip_grad_norm_: max_norm = 3
- batch size: 32


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
