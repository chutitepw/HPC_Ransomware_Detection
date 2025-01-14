# Hardware Performance Counter Ransomware Detection (POC)
#### This experiment demonstrates how performance counter data can be used to detect ransomware attacks with the help of various time series machine learning classifier models.

## Dataset
### Data Collection
The data is collected using the [K-LEB tool](https://github.com/chutitepw/K-LEB). K-LEB sets up and collects performance counter data from the processor in real-time and stores it in .csv format. For this experiment, K-LEB is collecting overall system data during normal system operations and when the system is under attack by ransomware. The data is collected in 10ms intervals to represent time series data. The hardware events used are listed below:

| Hardware Event* | Description | Event Number | Umask Value
| ------------- | ------------- | ------------- | ------------- |
| BR_RET  | Number of branch instructions retired.  | 0xc2 | 0x00
| INST_RET  | Number of instructions retired.  | 0xc0 | 0x00
| DCACHE_ACCESS | All Data Cache Accesses. | 0x29 | 0x07
| LOAD | Dispatch of a single op that performs a memory load. | 0x29 | 0x01
| STORE | Dispatch of a single op that performs a memory store. | 0x29 | 0x02
| MISS_LLC | L3 Misses by Request Type. | 0x9a | 0xff

*Different hardware events can be used but the performance may vary.

### Data File
In the data folder, there are data for training and testing for both benign and ransomware cases. 
| Files | Description 
| ------------- | ------------- |
| benign-*.csv  | File represents benign cases of the system. 
| ransom-*.csv  | Files represent different ransomware family's behaviors.

## Classifier
In this experiment, data classification is performed using different types of classifiers such as neural networks, and gradient-boosting models.
| Files | Description 
| ------------- | ------------- |
| cnn.py  |  Convolutional Neural Network (CNN) two-class classifier.
| lightgbm.py  | Light Gradient-Boosting Model (LightGBM) two-class classifier.
| lstm-anomaly.py | Long Short-Term Memory (LSTM) one-class anomaly detection classifier.
| lstm.py | Long Short-Term Memory (LSTM) two-class classifier.
| mlp.py | Multilayer perception (MLP) two-class classifier.
| xgboost.py | eXtreme Gradient Boosting (XGBoost) two-class classifier.

# Citing

For more technical details please refer to the following papers:
```
@ARTICLE{10208245,
  author={Woralert, Chutitep and Liu, Chen and Blasingame, Zander},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers}, 
  title={HARD-Lite: A Lightweight Hardware Anomaly Realtime Detection Framework Targeting Ransomware}, 
  year={2023},
  volume={70},
  number={12},
  pages={5036-5047},
  keywords={Ransomware;Monitoring;Behavioral sciences;Hardware;Servers;Registers;Operating systems;Performance evaluation;Semisupervised learning;Anomaly detection;Performance monitoring counters;semi-supervised learning;ransomware;anomaly detection;malware analysis},
  doi={10.1109/TCSI.2023.3299532}}
```
```
@inproceedings{10.1145/3696843.3696847,
author = {Woralert, Chutitep and Liu, Chen and Blasingame, Zander},
title = {Towards Effective Machine Learning Models for Ransomware Detection via Low-Level Hardware Information},
year = {2024},
isbn = {9798400712210},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3696843.3696847},
doi = {10.1145/3696843.3696847},
abstract = {In recent years, ransomware attacks have grown dramatically. New variants continually emerging make tracking and mitigating these threats increasingly difficult using traditional detection methods. As the landscape of ransomware evolves, there is a growing need for more advanced detection techniques. Neural networks have gained popularity as a method to enhance detection accuracy, by leveraging low-level hardware information such as hardware events as features for identifying ransomware attacks. In this paper, we investigated several state-of-the-art supervised learning models, including XGBoost, LightGBM, MLP, and CNN, which are specifically designed to handle time series data or image-based data for ransomware detection. We compared their detection accuracy, computational efficiency, and resource requirements for classification. Our findings indicate that particularly LightGBM, offer a strong balance of high detection accuracy, fast processing speed, and low memory usage, making them highly effective for ransomware detection tasks.},
booktitle = {Proceedings of the International Workshop on Hardware and Architectural Support for Security and Privacy 2024},
pages = {10–18},
numpages = {9},
keywords = {Performance Monitoring Counters, Supervised Learning, Ransomware},
location = {
},
series = {HASP '24}
}
```
