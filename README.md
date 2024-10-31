# WAKE: A Weakly-supervised Business Process Anomaly Detection Framework via a Pre-trained Autoencoder
This is the source code of our paper '[**WAKE: A Weakly-supervised Business Process Anomaly Detection Framework via a Pre-trained Autoencoder**](https://ieeexplore.ieee.org/abstract/document/10285076/)'.
![model](architecture.png)


## Requirements
- [PyTorch](https://pytorch.org)
- [NumPy](https://numpy.org)
- [scikit-learn](https://scikit-learn.org)
- [pm4py](https://pm4py.fit.fraunhofer.de/)


## Using Our Code
```
    python main.py --mode eval
```

Two modes have been implemented:

_eval_:  Utilizing the anomalous event logs located in the _**eventlogs**_ folder to obtain evaluation results (For reproducibility of the experiments).

_test_: Detecting anomalies in the event log with the 'xes' format and obtaining anomaly detection results (For practical application).

## Datasets
Eight commonly used real-life logs:

i) **_Billing_**: This log contains events that pertain to the billing of medical services provided by a hospital.

ii) **_Receipt_**: This log contains records of the receiving phase of the building permit application process in an anonymous municipality.

iii) **_Sepsis_**: This log contains events of sepsis cases from a hospital.

iv) **_Request_**: This log contains events related to Requests for Payment.

v) **_Prepaid_**: This log contains events related to prepaid travel costs.

vi) **_Loan_**: This log contains events related to the loan application process of a Dutch financial institute.

vii) **_Approval_**: This log contains information for loan and overdraft approvals processes from submission to eventual resolution (approval, cancellation, or rejection).

viii) **_Declaration_**: This log contains events related to domestic travel declarations.

Eight synthetic logs: i.e., **_Paper_**,  _**P2P**_, **_Small_**, **_Medium_**, **_Large_**, **_Huge_**, **_Gigantic_**, and **_Wide_**.

The summary of statistics for each event log is presented below:

| Log            | #Activities    | #Traces    | #Events         | Max trace length       | Min trace length    | #Attributes    | #Attribute values  |
|:--------------:|:--------------:|:----------:|:---------------:|:----------------------:|:-------------------:|:--------------:|:------------------:|
| Gigantic       | 76-78          | 5000       |  28243-31989    | 11                     | 3                   |  1-4           |  70-363            |
| Huge           | 54             | 5000       |  36377-42999    | 11                     | 5                   |  1-4           |  69-340            |
| Large          | 42             | 5000       |  51099-56850    | 12                     | 10                  |  1-4           |  68-292            |
| Medium         | 32             | 5000       |  28416-31372    | 8                      | 3                   |  1-4           |  66-276            |
| P2p            | 13             | 5000       |  37941-42634    | 11                     | 7                   |  1-4           |  39-146            |
| Paper          | 14             | 5000       |  49839-54390    | 12                     | 9                   |  1-4           |  36-128            |
| Small          | 20             | 5000       |  42845-46060    | 10                     | 7                   |  1-4           |  39-144            |
| Wide           | 23-34          | 5000       |  29128-31228    | 7                      |  5-6                |  1-4           |  53-264            |
| Approval       | 36             | 13087      | 262200          | 175                    | 3                   | 0              | 0                  |
| Loan           | 26             | 31509      | 1202267         | 180                    | 10                  | 1              | 149                |
| Declaration    | 17             | 10500      | 56437           | 24                     | 1                   | 2              | 9                  |
| Prepaid        | 29             | 2099       | 18246           | 21                     | 1                   | 2              | 10                 |
| Request        | 19             | 6886       | 36796           | 20                     | 1                   | 2              | 10                 |
| Billing        | 18             | 100000     | 451359          | 217                    | 1                   | 0              | 0                  |
| Receipt        | 27             | 1434       | 8577            | 25                     | 1                   | 2              | 58                 |
| Sepsis         | 16             | 1050       | 15214           | 185                    | 3                   | 1              | 26                 |

Logs containing 10% artificial anomalies are provided in the folder '**_eventlogs_**'.


## Experiment Results
_F−scores_ over synthetic logs where ’AD’ represents weakly supervised business process anomaly detection and ’IA’ represents interpretation of the cause of anomalies.

|         | Paper       | Paper       | P2P         | P2P         | Small       | Small       | Medium      | Medium      | Large       | Large       | Huge        | Huge        | Gigantic    | Gigantic    | Wide        | Wide         |
|:-------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|         | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**           |
| WAKE    | 0.965±0.021 | 0.780±0.014 | 0.953±0.013 | 0.733±0.016 | 0.969±0.011 | 0.764±0.014 | 0.920±0.015 | 0.749±0.015 | 0.930±0.024 | 0.788±0.017 | 0.917±0.022 | 0.784±0.007 | 0.909±0.024 | 0.727±0.024 | 0.936±0.009 | 0.728±0.019  |
| DAE     | 0.749±0.006 | 0.313±0.002 | 0.701±0.019 | 0.360±0.002 | 0.796±0.011 | 0.319±0.001 | 0.660±0.009 | 0.350±0.001 | 0.687±0.008 | 0.282±0.001 | 0.555±0.012 | 0.318±0.001 | 0.446±0.018 | 0.292±0.002 | 0.707±0.015 | 0.368±0.001  |
| VAE     | 0.765±0.012 | 0.291±0.001 | 0.496±0.007 | 0.327±0.004 | 0.723±0.010 | 0.309±0.002 | 0.566±0.021 | 0.327±0.002 | 0.704±0.023 | 0.260±0.002 | 0.426±0.017 | 0.276±0.003 | 0.311±0.013 | 0.266±0.003 | 0.520±0.017 | 0.345±0.001  |
| LAE     | 0.631±0.024 | 0.305±0.004 | 0.627±0.037 | 0.379±0.011 | 0.739±0.009 | 0.336±0.004 | 0.549±0.031 | 0.365±0.006 | 0.430±0.036 | 0.297±0.006 | 0.446±0.038 | 0.329±0.008 | 0.340±0.015 | 0.294±0.007 | 0.734±0.023 | 0.370±0.004  |
| BINet   | 0.632±0.084 | 0.596±0.039 | 0.643±0.074 | 0.591±0.025 | 0.667±0.071 | 0.615±0.026 | 0.604±0.060 | 0.589±0.015 | 0.633±0.080 | 0.575±0.040 | 0.608±0.069 | 0.599±0.023 | 0.555±0.054 | 0.613±0.018 | 0.601±0.070 | 0.582±0.017  |
| GAE     | 0.363±0.017 | -           | 0.511±0.015 | -           | 0.374±0.013 | -           | 0.284±0.010 | -           | 0.480±0.028 | -           | 0.296±0.007 | -           | 0.308±0.010 | -           | 0.560±0.027 | -            |
| DeepSAD | 0.710±0.107 | -           | 0.698±0.043 | -           | 0.778±0.079 | -           | 0.696±0.059 | -           | 0.654±0.077 | -           | 0.520±0.050 | -           | 0.567±0.047 | -           | 0.629±0.038 | -            |
| REPEN   | 0.682±0.037 | -           | 0.411±0.043 | -           | 0.744±0.027 | -           | 0.387±0.059 | -           | 0.546±0.057 | -           | 0.380±0.035 | -           | 0.287±0.026 | -           | 0.485±0.070 | -            |
| DevNet  | 0.794±0.053 | -           | 0.719±0.030 | -           | 0.812±0.021 | -           | 0.751±0.010 | -           | 0.771±0.024 | -           | 0.690±0.017 | -           | 0.686±0.016 | -           | 0.787±0.014 | -            |
| PReNet  | 0.695±0.035 | -           | 0.593±0.027 | -           | 0.779±0.026 | -           | 0.675±0.023 | -           | 0.590±0.034 | -           | 0.572±0.027 | -           | 0.657±0.020 | -           | 0.714±0.020 | -            |
| FEAWAD  | 0.801±0.039 | -           | 0.709±0.032 | -           | 0.861±0.026 | -           | 0.764±0.048 | -           | 0.701±0.045 | -           | 0.672±0.050 | -           | 0.674±0.043 | -           | 0.786±0.041 | -            |

_F−scores_ over real-life logs where ’AD’ represents weakly supervised business process anomaly detection and  ’IA’ represents interpretation of the cause of anomalies.

|         | Billing     | Billing     | Receipt      | Receipt     | Sepsis      | Sepsis      | Request     | Request     | Prepaid     | Prepaid     | Loan        | Loan        | Approval    | Approval    | Declaration | Declaration  |
|:-------:|:-----------:|:-----------:|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|         | **AD**          | **IA**          | **AD**           | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**          | **AD**          | **IA**           |
| WAKE    | 0.940±0.022 | 0.557±0.031 | 0.564±0.052  | 0.780±0.019 | 0.469±0.037 | 0.619±0.002 | 0.934±0.022 | 0.547±0.009 | 0.655±0.027 | 0.768±0.018 | 0.720±0.018 | 0.597±0.009 | 0.845±0.022 | 0.621±0.024 | 0.930±0.022 | 0.722±0.027  |
| DAE     | 0.583±0.022 | 0.492±0.008 | 0.478±0.010  | 0.337±0.002 | 0.307±0.003 | 0.338±0.001 | 0.756±0.002 | 0.409±0.002 | 0.577±0.000 | 0.222±0.003 | 0.313±0.002 | 0.252±0.001 | 0.521±0.012 | 0.446±0.005 | 0.747±0.004 | 0.420±0.006  |
| VAE     | 0.601±0.004 | 0.452±0.004 | 0.363±0.012  | 0.315±0.005 | 0.281±0.010 | 0.328±0.010 | 0.447±0.021 | 0.325±0.004 | 0.260±0.013 | 0.188±0.003 | 0.196±0.002 | 0.212±0.002 | 0.240±0.006 | 0.423±0.006 | 0.603±0.005 | 0.336±0.003  |
| LAE     | 0.701±0.015 | 0.514±0.011 | 0.419±0.034  | 0.295±0.009 | 0.280±0.008 | 0.255±0.016 | 0.715±0.028 | 0.520±0.009 | 0.496±0.023 | 0.261±0.016 | 0.199±0.001 | 0.261±0.004 | 0.483±0.027 | 0.523±0.024 | 0.709±0.026 | 0.570±0.040  |
| BINet   | 0.604±0.045 | 0.536±0.016 | 0.435±0.046  | 0.404±0.015 | 0.342±0.017 | 0.181±0.022 | 0.640±0.017 | 0.511±0.007 | 0.650±0.023 | 0.616±0.019 | 0.494±0.036 | 0.493±0.016 | 0.596±0.132 | 0.583±0.037 | 0.517±0.026 | 0.668±0.010  |
| GAE     | 0.275±0.016 | -           | 0.307±0.029  | -           | 0.231±0.003 | -           | 0.360±0.000 | -           | 0.288±0.007 | -           | 0.182±0.000 | -           | 0.346±0.018 | -           | 0.355±0.002 | -            |
| DeepSAD | 0.804±0.009 | -           | 0.293±0.047  | -           | 0.287±0.019 | -           | 0.611±0.056 | -           | 0.556±0.015 | -           | 0.253±0.005 | -           | 0.494±0.030 | -           | 0.673±0.068 | -            |
| REPEN   | 0.542±0.022 | -           | 0.366±0.035  | -           | 0.244±0.012 | -           | 0.573±0.042 | -           | 0.465±0.042 | -           | 0.189±0.001 | -           | 0.240±0.003 | -           | 0.576±0.017 | -            |
| DevNet  | 0.743±0.004 | -           | 0.528±0.027  | -           | 0.297±0.016 | -           | 0.850±0.011 | -           | 0.595±0.041 | -           | 0.269±0.005 | -           | 0.619±0.005 | -           | 0.863±0.010 | -            |
| PReNet  | 0.776±0.004 | -           | 0.512±0.061  | -           | 0.315±0.014 | -           | 0.728±0.017 | -           | 0.577±0.017 | -           | 0.260±0.005 | -           | 0.657±0.004 | -           | 0.865±0.011 | -            |
| FEAWAD  | 0.780±0.018 | -           | 0.457±0.044 | -           | 0.328±0.024 | -           | 0.815±0.020 | -           | 0.563±0.032 | -           | 0.300±0.003 | -           | 0.553±0.035 | -           | 0.847±0.013 | -            |


## To Cite Our Paper
```
@ARTICLE{10285076,
  author={Guan, Wei and Cao, Jian and Zhao, Haiyan and Gu, Yang and Qian, Shiyou},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={WAKE: A Weakly Supervised Business Process Anomaly Detection Framework via a Pre-Trained Autoencoder}, 
  year={2024},
  volume={36},
  number={6},
  pages={2745-2758},
  keywords={Anomaly detection;Business;Feature extraction;Hidden Markov models;Probabilistic logic;Generators;Data models;Process mining;weakly supervised anomaly detection;deep learning;recurrent neural networks;autoencoder},
  doi={10.1109/TKDE.2023.3322411}}
```