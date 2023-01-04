# StarGraph + Text for [OGB WikiKG 2](https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2)

Existing methods for the challenge only use information in the provided KG, but external data such as entity descriptions, entity types, and visual information are ignored. In this report, we utilize entity descriptions of Wikidata to enhance the KG representation. Specifically, we downloaded and parsed the JSON dump of all Wikibase entries for Wikidata generated on November 30, 2015 from https://archive.org/details/wikidata-json-20151130. For each entity, we embed the description text with Sentence-BERT (the best performing all-mpnet-base-v2 model), and then initialize the entity embeddings in existing methods with our text embeddings. We call our method X + Text, where X is the base method. In this report, we use StarGraph + TripleRE as the base method.

|Method|Test MRR|Validation MRR|#Params|
|-|-|-|-|
|**StarGraph + TripleRE + Text **|**0.7305 ± 0.0010**|**0.7442 ± 0.0006**|1,927,395,330|
|StarGraph + TripleRE|0.7201 ± 0.0011|0.7288 ± 0.0008|86,762,146|



+ This is the code to run StarGraph + Text on the OGB WikiKG 2 dataset. 
The code is based on [StarGraph repo](https://github.com/hzli-ucas/StarGraph) and [NodePiece repo](https://github.com/migalkin/NodePiece/tree/main/ogb).
+ This entity and relation descriptions are in dataset/ogbl_wikikg2/mapping/. 

## Running
1. Install the requirements from the `requirements.txt`

2. Prepare the file storing the subgraphs as follows:  

a. Download the file of anchors using the `download.sh` script, provided by [NodePiece](https://github.com/migalkin/NodePiece/blob/main/ogb/download.sh) 

&emsp;&emsp; b. Generate the file of neighbors by running `python create_nborfile.py`

&emsp;&emsp; c. Unzip dataset/ogbl_wikikg2/mapping/nodeidx2entityid_des.txt.zip

&emsp;&emsp; d. Generate the entity text embeddings by running `python sent_emb.py`

3. Run the `run_ogb.sh` script to reproduce the results of **StarGraph + TripleRE + Text** reported above

## Citation
If you find this work useful, please consider citing the paper:
```
@article{yao2023ogb,
  title={Technical Report for OGB Link Property Prediction: ogbl-wikikg2},
  author={Yao, Liang and Peng, Jiazhen and Liu, Qiang and Cai, Hongyun and Ji, Shengong and He, Feng and Cheng, Xu},
  year={2023}
}
```
