## Multi-Source Domain Adaptation with Mixture of Experts
Code and data for the EMNLP 2018 paper: [Multi-Source Domain Adaptation with Mixture of Experts](https://arxiv.org/abs/1809.02256)

### Running

```
cd msda-src
# example script for training uni-MS (baseline)
./train_unified.sh
(run "python amazon-chen/senti_unified.py -h" for full options)

# example script for training MoE
./train_moe.sh
(run "python amazon-chen/senti_moe.py -h" for full options)
```
(to be updated to latest versions of pytorch)

Note: The official Chen12 dataset doesn't contain a dev split. To perform hyper-parameter selection, you should create multiple folds by randomly splitting dev sets (1/10) from the (multi-source) training data as a means of cross-validation. They follow the same naming convention of `${domain}_dev.svmlight` under the same directory of the training and test sets.

### Dependencies
* Pytorch 0.3/0.4
* sklearn
* termcolor

### References

```
@InProceedings{guo2018multi,
  author = "Guo, Jiang and Shah, Darsh J and Barzilay, Regina",
  title = "Multi-Source Domain Adaptation with Mixture of Experts",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  pages = "4694--4703",
  location = "Brussels, Belgium",
  url = "http://aclweb.org/anthology/D18-1498"
}
```

### Contact
Please create an issue or email to [jiang_guo@csail.mit.edu](mailto:jiang_guo@csail.mit.edu) should you have any questions, comments or suggestions.

