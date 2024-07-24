# Federated IDS Models

This is the code for my research masters, which involves testing the effectiveness of different models when being used for federated IDS systems. The main goal of this is to gain an idea of which models are good for IoT security approaches. These are designed to be used with the CSE IoT2023 dataset, but any should suffice so long as the data is adequately cleaned **and made dual class for SVM**.

This does not include a lot of the preprocessing steps which will be included in the thesis and corresponding papers.

## Support Vector Machine
This includes an implementation (maybe the first) of flower-federated support vector machine. This requires some more data preprocessing than other models. Like the others it needs to be preprocessed, also it needs to be made dual class. This is important unless you have hundreds of gigabytes of RAM lying around. 

The max_iter is set at 50 in this implementation, for experimental purposes it should be run at 10,000+, this still loses some resolution but not enough to make it not worthwhile.

## Required Libraries

The following python libraries need to be installed:

 - TQDM
 - Sklearn
 - Flower (flwr)
 - Numpy
 - Pandas

```
pip install flwr scikit-learn numpy pandas tqdm
```
