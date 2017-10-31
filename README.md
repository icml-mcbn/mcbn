# Bayesian Uncertainty Estimation for Batch Normalized Deep Networks, ICLR 2018 Conference-Track Submission
### The codes associated with generating the results of the above paper are released in this repository for reproducability purposes.

#### Bayesian SegNet with MCBN
In this experiment we generate qualitative results when using MCBN uncertainty estimation for Bayesian SegNet[1]. Two other repositories are required for reproducing the figure. One is a caffe fork modified for SegNet which can be found here: https://github.com/TimoSaemann/caffe-segnet-cudnn5  The other is the SegNet: https://github.com/TimoSaemann/caffe-segnet-cudnn5

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




