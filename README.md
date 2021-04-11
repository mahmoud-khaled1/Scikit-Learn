# Scikit-Learn

![SKLearn](https://user-images.githubusercontent.com/43557035/95685305-478a3680-0bf7-11eb-8a4d-c63104eb2935.jpg)

## Scikit-learn: is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

# Stages Of A Machine Learning Project:
## 1-Problem Definition 
* Problem definition is the initial stage of a Computer Vision/ML project, and it focuses on gaining an understanding of the problem poised to be solved by applying ML.
## 2- Data Aggregation / Mining / Scraping
* Data is the fuel for an ML/CV application. Data aggregation is a crucial step that sets a precedent for the effectiveness and performance of the trained model.
## 3-Data Preparation / Preprocessing / Augmentation
Preprocessing steps for data are based mainly on the model input requirements. Refer back to the research stage and recall input parameters and requirements that the selected model / neural network architecture requires.
* 3.1- Attach DataSet
* 3.2- Data Cleaning
* 3.3- Feature Selection 
* 3.4- Data Scaling 
* 3.5- Data Split 
## 4- Model Implementation
Choice best algorithm based on your problems (Linear Regression - Logistic Regression(Classification) - Neural Network - SVC - SVR - KMeans-KNN - Decision Tree..etc).
## 5- Training Your Model 
The training data delivered from the previous Data stages are utilized within the training stage. The implementation of model training involves passing the refined aggregated training data through the implemented model to create a model that can perform its dedicated task well.
When conducting training, it is vital to ensure that metrics are recorded of each training process and at each epoch. The metrics that are generally collected are the following:
* Training accuracy
* Validation accuracy
* Training Loss
* Validation Loss
By visualizing the training metrics, it is possible to identify some common ML model training pitfalls, such as underfitting and overfitting.
* Underfitting: This occurs when a machine learning algorithm fails to learn the patterns in a dataset. Underfitting can be fixed by using a better algorithm or model that is more suited for the task. Underfitting can also be adjusted fixed by recognizing more features within the data and presenting it to the algorithm.
* Overfitting: This problem involves the algorithm predicting new instances of patterns presented to it, based too closely on instances of patterns it observed during training. This can cause the machine-learning algorithm to not generalize accurately to unseen data. Overfitting can occur if the training data does not accurately represent the distribution of test data. Overfitting can be fixed by reducing the number of features in the training data and reducing the complexity of the network through various techniques.
## 6- Evaluation
At this stage, you should have a trained model and are ready to conduct evaluation techniques on its performance.
For evaluation, we utilize a partition of the refined data, usually referred to as the ‘test data’. The test data have not been seen during the model during training. They are also representative of examples of data that are expected to be encountered in practical scenarios.
* Confusion matrix (error matrix)
* Precision-Recall
## 7- Parameter tuning and Inference
Parameter tuning is the process of model refinement that is conducted by making modifications to hyperparameter values. The purpose of parameter tuning is to increase the model performance, and this correlates to improvements in evaluation results.
Once hyperparameters are tuned and new values are selected, training and evaluation commence again.
## 8- Model Conversion to appropriate
Once we have our refined model, we are ready to place it on devices where it can be utilized.
Model conversion is a step that is required when developing models that are to be used within edge devices such as mobile phones or IoT devices.
## 9- Model Deployment
Deploying our final trained model is the last step within all the identified stages. Integrating our model within a broader ecosystem of application or tool, or simply building an interactive web interface around our model, is an essential step of model deployment.

