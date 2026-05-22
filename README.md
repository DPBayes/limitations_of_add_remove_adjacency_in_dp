## Official code for the paper: ``Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy``

### Dependencies:

The following modules are required to run the code:
 * Python 3.8 or greater
 * PyTorch 1.11 or greater (https://pytorch.org/)
 * Opacus 1.3 or greater (https://opacus.ai/#quickstart)
 * prv_accountant 0.2.0 or greater (https://github.com/microsoft/prv_accountant)
 * scipy
 * statsmodel
 * pandas
 * numpy
 * pickle

### Source Code Libraries:

We adapt/ use the code available on the following open source code libraries, some of which we have modified:
* TIMM (for the PyTorch VIT-B implementation): https://github.com/rwightman/pytorch-image-models Copyright 2020 Ross Wightman.
* https://github.com/cambridge-mlg/dp-few-shot (for fine-tuning ViT-B-16 model using CIFAR10).
* https://github.com/microsoft/MICO (for training models from scratch using Purchase100).

### Caching Feature Representations:

It is more computationally efficient to cache feature representations and load them for training models when only the *final* linear layer of the model needs to be trained.

Use `feature_space_cache/map_to_feature_space.py` to save representations obtained from pre-trained models (ViT-B/R-50) from datasets in the feature dimension. This has to be done only once for each dataset.

### Running Auditing Algorithms:

To audit DP models fine-tuned with CIFAR10 in the substitute-adjacency threat model with various canaries:

* **Section [3.1]**: Auditing Using Crafted Dataset Canaries: Run ```src/auditing_with_worst_case_datasets.ipynb```
* **Algorithm [2]**: Auditing Using Gradient-Space Canaries: Run ```src/auditing_with_canary_gradient.py```
* **Algorithm [3]**: Auditing Using Crafted Input Canary: Run ```src/auditing_with_canary_input.py``` 
* **Algorithm [4]**: Auditing Using Crafted Mislabeled Canary: Run ```src/auditing_with_canary_label.py```
* **Algorithm [5]**: Auditing Using Adversarial Natural Canary:  Run ```src/auditing_with_natural_sample.py```
* **Section [6.2.3]**: To audit MLP trained from scratch using Purchase100: Run ```src/auditing_models_trained_from_scratch.py```.

### Plotting Auditing Results: 

We provide sample code in ```src/plotting_audit_results.ipynb``` to plot the auditing results.


If you use this work, please cite:

```bibtex
@article{pradhan2025membershiplimitationsaddremoveadjacency,
      title={Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy}, 
      author={Gauri Pradhan and Joonas Jälkö and Santiago Zanella-Bèguelin and Antti Honkela},
      year={2025},
      eprint={2511.21804},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2511.21804}, 
}
