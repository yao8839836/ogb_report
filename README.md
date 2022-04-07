# Feature propagation for link prediction
The ogbl-collab and ogbl-citation2 are two datasets for link prediction.
The challenge leaderboard can be checked at: https://ogb.stanford.edu/docs/leader_linkprop/.
We apply feature propagation to solve this challenge and this repo contains our code submission.
The techniqual report can be checked at [ogb_report.pdf](https://github.com/yao8839836/ogb_report/blob/main/ogb_report.pdf)

## Requirements
  Install base packages:
    ```
    Python==3.6
    Pytorch==1.7.1
    pytorch_geometric==2.0.1
    ogb==1.3.2
    ```

## Results on OGB Challenges
Running the default code 10 times, here we present our results on the ogbl-collab and ogbl-citation2.

|   Method    | ogbl-collab (Hits@50)      | ogbl-citation2 (MRR)   |
| ---------- | :-----------:  | :-----------: |
| PLNLP | 0.7046 ± 0.0040  | -- |
|  PLNLP + SIGN | 0.7087 ± 0.0033  | --  |
|  MLP | 0.1991 ± 0.0170  | 0.2900 ± 0.0018 |
|  MLP + SIGN | 0.2839 ± 0.0127  | 0.3224 ± 0.0017 |

## Training Process for ogbl-collab

1) PLNLP as backbone
```
python plnlp_sign.py --data_name=ogbl-collab  --predictor=DOT --use_valedges_as_input=True --year=2010 --train_on_subgraph=True --epochs=800 --eval_last_best=True --dropout=0.3 --gnn_num_layers=1 --grad_clip_norm=1 --use_lr_decay=True --random_walk_augment=True --walk_length=10 --loss_func=WeightedHingeAUC --data_path=./dataset
```
2) MLP as backbone
```
python mlp_collab_sign.py
```


## Training Process for ogbl-citation2

```
python mlp_citation2_sign.py
```


Reference 
---------
- https://ogb.stanford.edu/
- https://github.com/snap-stanford/ogb
- https://github.com/zhitao-wang/PLNLP
- Pairwise Learning for Neural Link Prediction (https://arxiv.org/pdf/2112.02936.pdf)

