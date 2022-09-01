# Functional Tissue Segmentation
### Motivation 
---
The goal of this competition is to identify the locations of each functional tissue unit (FTU) in biopsy slides from several different organs.
The underlying data includes imagery from different sources prepared with different protocols at a variety of resolutions, reflecting typical 
challenges for working with medical data.

This competition uses data from two different consortia, the Human Protein Atlas (HPA) and Human BioMolecular Atlas Program (HuBMAP). 
The training dataset consists of data from public HPA data, the public test set is a combination of private HPA data and HuBMAP data, and 
the private test set contains only HuBMAP data. Adapting models to function properly when presented with data that was prepared using a different
protocol will be one of the core challenges of this competition. While this is expected to make the problem more difficult, developing models that generalize 
is a key goal of this endeavor.

### Data
---
- [HuBMAP](https://www.kaggle.com/competitions/hubmap-organ-segmentation/data) Dataset obtained from kaggle by National Institutes of Health (NIH), HuBMAP and Indiana University’s.
