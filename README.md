# Augmenting Graph Inductive Learning Model With Topographical Features
Knowledge Graph (KG) completion aims tofind the missing entities or relationships ina knowledge graph. Although many ap-proaches have been proposed to constructthe complete KGs, recently, graph embed-ding methods have gained massive attention.These methods performed well in transduc-tive settings, where the entire collection of en-tities must be known during training. How-ever, these embedding methods have somelimitations in explicitly capturing the rela-tional semantics when new entities are addedto KGs over time. Recently GraIL proposeda Graph Neural Network (GNN) based re-lation prediction method to learn such rela-tional semantics even if the entities were un-seen during training. However, since GraIL operates strictly on subgraphs, there is noextra information from the whole KG avail-able to the GNN during training. So, thispaper proposes a framework that improvesthe GraIL method by incorporating topolog-ical features of each entity during feature ex-traction. Experiments on standard datasetsdemonstrate that our augmentation of GraILdelivers more accuracy in relations predictionin both transductive and inductive settings.

The code provided is a modified version of the GraIL implementation by K. Teru. The implemenation of our specific improvement can be found in the subgraph_extraction/datasets.py file. This is where we implement the PLACn feature extraction. The following readme is also a modified version of the GraIL readme since running the commands are identical.

## Requirements
All the required packages can be installed by running ```pip install -r requirements.txt.```

## Inductive relation prediction experiments
All train-graph and ind-test-graph pairs of graphs can be found in the ```data``` folder. We use WN18RR_v1 as a runninng example for illustrating the steps.

## AGIL
To start training a AGIL model, run the following command. ```python3 train.py -d WN18RR_v1 -e AGIL_wn_v1```

To test AGIL run the following commands.

- ```python3 test_auc.py -d WN18RR_v1_ind -e AGIL_wn_v1```

- ```python3 test_ranking.py -d WN18RR_v1_ind -e AGIL_wn_v1```

The trained model and the logs are stored in ```experiments``` folder. Note that to ensure a fair comparison, we test all models on the same negative triplets. In order to do that in the current setup, we store the sampled negative triplets while evaluating AGIL and use these later to evaluate other baseline models.

## Transductive experiments
The full transductive datasets used in these experiments are present in the data folder.
