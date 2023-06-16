# Small Data Big Dreams
## Intro/Background - Gabriel

Data augmentation and transfer learning are vital for leveraging machine learning on small datasets. These techniques mitigate overfitting and improve model generalizability by artificially expanding the dataset and leveraging pre-existing knowledge, respectively. 

In this exploration, we'll delve into the application of these and similar methodologies for image classification and tabular regression problems on small datasets, presenting practical insights and challenges encountered in implementing these techniques.

## Problem Statement - Lucy

The motivation of this project is to experiment with various dataset sizes and methodologies for data augmentation and transfer learning processes. Overfitting, limited dataset sizes, and overall accuracy in machine learning are still areas rife with potential for new development today, and the ability to artificially expand training sets from existing data offers an efficient breakthrough in the field. However, data augmentation today is still limited by the effectiveness of current approaches.

## Methods - Hannah

We are going to explore different ways to deal with small datasets.

### Data augmentation:
Data augmentation methods, such as random croppings, rotations, and adding Gaussian noise, are ways to deal with small datasets. We will benchmark each data augmentation method as well as combinations of them to find the optimal combo.

### Generative models for data synthesis:
Synthetic data as a supplement to real data is another way to deal with small datasets. Specifically, we will test the GAN and Diffusion models on their efficacy in generating synthetic data.
- GAN: We’ll use GAN on existing datasets to generate synthetic data.
- Diffusion model: We’ll use textual inversion on pre-trained diffusion models to engineer a prompt for our dataset, and use this prompt as a condition to generate synthetic data from the pre-trained diffusion model.
### Transfer learning: 
We’ll further explore how we can fine-tune existing large models for small dataset classification tasks.
- Changing all parameters of the model is probably unrealistic given our current devices.
- Use LoRA to fine-tune the classification model.
- Add layers on top of the pre-trained model and freeze other layers.
## Potential Results/Discussion - Hyuk

We'll use several metrics to gauge the success of our data augmentation techniques. The Fowlkes-Mallows index gauges the similarity between synthetic and original data, with a higher score signaling better augmentation. The AUC-ROC, an evaluation measure for classification problems, plots the True Positive Rate against the False Positive Rate. We anticipate improved scores with synthetic data. For multi-class models, multiple AUC-ROC curves will be generated. In tabular regression tasks, we'll use RMSE and MAE, metrics that quantify prediction deviations from actual values, thus offering a holistic view of our prediction accuracy. We aim for these scores to also improve with the use of synthetic data. After the use of data augmentation, we will utilize two main scoring metrics to determine the effectiveness of the synthetic data. First, the Fowlkess-Mallows Measure utilizes the following equation:

