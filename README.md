# Supervised Image Super Resolution and Denoising

## Team Members:
- Andrew Chen (aachen3)
- Patrick Cole (pacole2)

## Implemenation History
Initially we were creating models based off of papers that dealt with either denoising or super resolution. Our best models seemed to be based off of SRCNN (Super Resolution CNN) or something that had to do with Subpixel upscaling. With these models however our best scores were only on the border of the A baseline (~8100 rmse).

In lecture Prof. Koyejo suggested we think simpler models and that something from one of the first lectures could provide sufficiently good results due to the simplicity of the nature of xray images. From this Andrew suggested KNN as our new starting approach. We first started off with mapping the nearest xray from the training images to the test images based off of the euclidean distance from the input images. This suprisingly gave us around ~8063 rmse. Next we broke the implementation up into patches and compared each patch. Starting off with 8 patches and then moving up to 256 patches. Using 256 patches we were able to get an rmse ~5890. This result seemed really good considering the performance of other methods we had tried, so we didn't try increasing the number of neighbors to use to try to better our results. We also notices that 256 patches seemed to be the sweet spot because increasing anymore gave us a determental effect.

## Repository Guide
In the [`knn`](knn/) directory, we have our latest implementation that we are using as our final submission.

In [`Andrew`](Andrew/) directory, it contains all of the models that Andrew attempted throughout the process before reaching KNN.

In [`Patrick`](Patrick/) directory, it contains all of the models that Patrick attempted throughout the process before reaching KNN.

There are also a list of [papers](papers.md) that we used for inspiration, and accuracy_logs that show a list of how our models were performing given various hyperparameters.

[Andrew's accuracy_log](Andrew/accuracy_log.csv)

[Patrick's accuracy_log](Patrick/accuracy_log.csv)

## Usage Guide
To run the knn model, enter "python3 knn/knn.py" in the command line. The program assumes that the training and testing data is located in the "xray" directory.

To run either Andrew's or Patrick's implementations, enter the following in the command line:
"python3 <Andrew/Patrick>/main.py --train --test --model <model_name> --batch <batch_size> --epochs <epochs> --lr <learning_rate> --validation <validation_percentage>".
It may run with only --train to train and/or only --test to test. Andrew's implementations assume that the data is in the "xray" directory while Patrick's implementation assumes that the data is in the "xray_images" directory.
