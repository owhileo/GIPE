<h1 align="center"> On the Gini-impurity Preservation 
    For Privacy Random Forests </h1>
Python implementation of the NeurIPS 2023 paper: [On the Gini-impurity Preservation For Privacy Random Forests](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8d6b1d775014eff18256abeb207202ad-Abstract-Conference.html).



## Running the Code

### Our Gini-impurity preserving encryption

-  Directly run the ***GIPE_minmax.py*** and get the encrypted dataset

### The encrypted DT and RF
(1) ***EncryptRandomForest.py*** can train the random forest by the sklearn after we encrypt the feature space by our Gini-impurity preserving encryption
(2) ***EncryptDecisionTree_CKKS.py*** can train each tree in the random forest while the feature space is encrypted by our  Gini-impurity preserving encryption and label space is encrypted by the CKKS ( HElib library)

### Others
 ***security.py*** : This code gives the calculation of the bitwise leakage matrix for our GIPE method



## BibTeX

```bib
@inproceedings{xie2024gini,
      title={On the Gini-impurity Preservation For Privacy Random Forests}, 
      author={Xie, XinRan and Yuan, Man-Jie and Bai, Xuetong and Gao, Wei and Zhou, Zhi-Hua},
      booktitle = {Advances in Neural Information Processing Systems},
      pages = {45055--45082},
      volume = {36},
      year = {2023}
}
```