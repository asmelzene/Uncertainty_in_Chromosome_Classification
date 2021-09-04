# Uncertainty_in_Chromosome_Classification

Chromosome classification is one of the essential tasks in Karyotyping to diagnose genetic abnormalities like some types of cancers and down syndrome.  Deep Convolutional Neural Networks have been widely used in this task, and the accuracy of classification models is exceptionally critical to such sensitive medical diagnoses.

However, it is not always possible to meet the expected accuracy rates for diagnosis. So, it is vital to tell how certain or uncertain a model is with its decision. Research has been conducted to estimate the uncertainty by Bayesian methods and non-Bayesian Neural Networks, while little is known about the quality of uncertainty estimations. In our work, we use two metrics, entropy, and variance, as uncertainty measurements. Moreover, three additional metrics, fail rate, workload, and tolerance range, are used to measure uncertainty metrics’ quality. Four different non-Bayesian methods: deep ensembles, snapshot ensembles, Test Time Augmentation (TTA), and Test Time Dropout (TTD), are used in experiments. 

A negative correlation is observed between the accuracy and the uncertainty estimation; the higher the accuracy of the model, the lower the uncertainty. Densenet121 with deep ensembles gives the best outcomes in our uncertainty estimations. Variance leaves entropy behind as an overall comparison as an uncertainty metric.

This repository includes codes related to the above-mentioned project.
