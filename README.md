# Bayesian Uncertainty Estimation for Batch Normalized Deep Networks, ICLR 2018 Conference-Track Submission
### The codes associated with generating the results of the above paper are released in this repository for reproducability purposes.

#### MCBN Uncertainty quality evaluation on regression datasets
**code/mcbn:** This part of the code runs the uncertainty quality evaluation (normalized CRPS and PLL) in Table 2 of the paper. In addition to a tensorflow implementation of MCBN, this includes an implementation of Monte Carlo Dropout (MCDO) by Gal & Ghahramani (see https://github.com/yaringal/DropoutUncertaintyExps). The evaluation datasets are included in the repository and are from the [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) and the [Delve Datasets](http://www.cs.toronto.edu/~delve/data/datasets.html).

The full experiment consists of running 7 jupyter notebooks in sequence. This can be executed in two ways:

A. Using docker. This allows for running one of the datasets with one seed to separate test data from CV and training data. In the experiment root (code/mcbn), this is executed as follows:
   ```
   make build
   make run dataset=bostonHousing seed=1
   ```
   This runs the entire test procedure for this particular dataset and split (datasets are in the data/ dir). This is convenient to distribute the work over many machines. In the paper we used seeds ranging from 1 to 5 for all datasets.

B. By running each of the notebooks 01-07 in sequence manually. This allows for multiple datasets to be evaluated at once for one specific seed (see configuration in setup.yml below). This is executed as follows: 
   ```
   virtualenv env
   source env/bin/activate
   pip install -r requirements.txt
   pip install scikit-optimize==0.4
   jupyter notebook
   ```
   After this you may run the notebooks 01-07 in sequence.

Method A) overrides the datasets and split_seed parameters in _setup.yml_. This file specifies test configurations and by default reflects the configuration in the paper.

To track progress, tail the file _evaluation.log_ (created in the experiment root). Also, check the evaluations/ dir (created in the experiment root) which contains incremental results.

The main output is a csv file: _final_results.csv_, placed in the experiment root which contains test CRPS, PLL, RMSE, Optimal CRPS and Optimal PLL metrics for each test. The directory evaluations/test/ will also contain test set predictions and predicted variance for the different models, such that the plots in Figure 2 (and 5 and 6 in the appendix) can be produced.

We use [properscoring](https://pypi.python.org/pypi/properscoring) to calculate CRPS and [scipy](https://www.scipy.org/) and [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) for numerical optimization. 

#### Bayesian SegNet with MCBN
**code/segnet:** In this experiment we generate qualitative results when using MCBN uncertainty estimation for Bayesian SegNet[1]. Two other repositories are required for reproducing the figure. One is a caffe fork modified for SegNet which can be found here: https://github.com/TimoSaemann/caffe-segnet-cudnn5  The other is the SegNet: https://github.com/TimoSaemann/caffe-segnet-cudnn5

Install a working version of Bayesian SegNet and it would be straightforward then to reproduce our figure with the provided scripts.

As our base network we used the CamVid pretrained network for Bayesian SegNet, which can be downloaded from the model zoo: https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_zoo.md

The caffe deploy prototxt we used turns off all the dropout random sampling to turn off MCDO:

```
sample_weights_test: false
```
But turns a few BN layers into re-estimating its parameters (mean/variance) during test inference by removing the following lines from some BN blocks of the graph definition,

```
bn_mode: INFERENCE
```
After a successful installation of Bayesian SegNet, the following command along with our provided script can be used to produce similar figures to the one included in the paper:

```
python /SegNet/Scripts/test_bayesian_segnet_mcbn_paperResults.py --model /SegNet/Example_Models/bayesian_segnet_camvid_deploy_mcbn_select.prototxt --weights /SegNet/Example_Models/bayesian_segnet_camvid.caffemodel --colours /SegNet/Scripts/camvid11.png --testdata /SegNet/CamVid/test.txt --traindata /SegNet/CamVid/train.txt
```

**NOTE.**  *it should be noted that we have used the pretrained SegNet models for MCBN which is suboptimal. Specifically, it seems the pretrained models have been trained with a very low batch-size that can affect the MCBN which is directly related to batch size for randomization.*




