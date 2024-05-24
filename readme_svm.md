
# Glioma Diagnosis between High Grade and Low Grade

This project focuses on analyzing brain MRI images from the BRATS2020 dataset. Radiomic features were extracted from these images, specifically targeting regions with tumor masks. These features were utilized to train a Support Vector Machine (SVM) model for classification tasks.


## Data Partition

Here we got 369 data for each patients mri features with its Glioma grade Label whether its HGG or LGG.

First we are turning our String Label to Int Labels with label enncoder from sci kit, since str values when turned to represent data would be represented in float values. Since floats arent accurate the data points would be varying with each even for the same label.

After that we are taking 10 of each HGG and LGG for our hidden test.

From the rest of the data, we can see a class imbalance where LGG is the minority. We used some techniques available to overcome this,

1. SMOTE Oversampling:
    - Here we used oversampling method from inbuilt functionns from sci kit 
    to create synthetic dataof the mionrity class.
    
2. Stratified K- Fold:
    - with the train_test_split() we can turn on the stratify
    flag, unlike traditional K fold, stratified makes sure that
    depending on how manny folds we want, each fold will contain
    a balanced class labels .

3. Class Weight Balance:
    - This flag is turned on our SVM model initialization. This
    assigns a higher weight to the minority class, so the function 
    is more inclined towards the minority class.

We, eventually settled down with Class Weight Balance since all of it gave an almost equal level of accuracy


## Features Used

We are using the radiometric features extracted using the radiomics library. 
We are taking enntire 3d volume of brain for each channel and use the combined mask to pass into the feature extraction method of radiomics.From there we analyse repeatibilty scores to check out the variance  from repeated measurement or feature.
Then we take top 10 features from shape, intensity and texture. The following is the list of features employed in our analysis:

| Feature Category       | Feature Name                                 |
|------------------------|----------------------------------------------|
| **Volume**             | Volume                                       |
| **Shape Features**     | original_shape_Elongation                    |
|                        | original_shape_Flatness                      |
|                        | original_shape_LeastAxisLength               |
|                        | original_shape_MajorAxisLength               |
|                        | original_shape_Maximum2DDiameterColumn       |
|                        | original_shape_Maximum2DDiameterRow          |
|                        | original_shape_Maximum2DDiameterSlice        |
|                        | original_shape_Maximum3DDiameter             |
|                        | original_shape_MeshVolume                    |
|                        | original_shape_MinorAxisLength               |
| **First-order Statistics** | original_firstorder_90Percentile        |
|                        | original_firstorder_Entropy                  |
|                        | original_firstorder_InterquartileRange       |
|                        | original_firstorder_Maximum                  |
|                        | original_firstorder_MeanAbsoluteDeviation    |
|                        | original_firstorder_Minimum                  |
|                        | original_firstorder_Range                    |
|                        | original_firstorder_RobustMeanAbsoluteDeviation |
|                        | original_firstorder_RootMeanSquared          |
|                        | original_firstorder_Uniformity               |
| **GLCM (Gray Level Co-occurrence Matrix)** | original_glcm_Idmn   |
|                        | original_glcm_Idn                            |
|                        | original_glcm_Imc1                           |
| **GLDM (Gray Level Dependence Matrix)** | original_gldm_DependenceEntropy |
|                        | original_gldm_DependenceNonUniformity        |
|                        | original_gldm_DependenceNonUniformityNormalized |
| **GLRLM (Gray Level Run Length Matrix)** | original_glrlm_RunEntropy |
|                        | original_glrlm_ShortRunEmphasis              |
| **GLSZM (Gray Level Size Zone Matrix)** | original_glszm_SmallAreaEmphasis |
|                        | original_glszm_ZoneEntropy                   |



## Model Performance

### Training and Validation
The SVM model with an RBF kernel (`C = 1.0`, `gamma = 1e-07`) was trained using 10-fold cross-validation. The average performance metrics across the folds were:

- **Validation Accuracies**: 0.815, 0.812, 0.822, 0.828, 0.822, 0.821, 0.822, 0.828, 0.822, 0.825
- **Accuracy Scores**: 0.829, 0.857, 0.857, 0.829, 0.829, 0.800, 0.829, 0.829, 0.800, 0.765

### Hidden Test Set
The final model achieved an accuracy of 0.5 on the hidden test set, indicating areas for improvement with unseen data.



## Feature Selection Considerations and Discussion

When working with radiomic features, ensuring their repeatability is important to maintain consistency and not much variance across different instances. However, just depending on repeatability of a feature is not advisable. Some features might not be relevant in classifying the labels, and more of the features which might not be related at all would end up making the model complex as well. Therefore criteria such as relevance and redundancy reduction are important to be considered as well. This would ensure we have relevant data to give the best possible prediction while making a stable robust model.


## Challenges Encountered

Throughout the classification process, several challenges were encountered. These challenges included:

1. **Class Imbalance**: Dealing with class imbalance, particularly when LGG class is significantly underrepresented as minonrity, posed challenges in model training and evaluation. Techniques such as oversampling methods or class weighting wereused to figure the best outcome.

2. **Feature Selection**: Selecting the most relevant and informative features from a large set of radiomic features. Making sure to reduce feature dimensionality while making good predictive outcome was challenging.

3. **Model Interpretability**: SVM models are good at classification, yet it was hard to find a good decision boundary to represent the underlying data pattern.

4. **Hyperparameter Tuning**: Having to search for the best parameters with GridSearch for the SVM model with different kernels, Gamma values was tricky. Main challenge was time taken to find the best parameters and best model.

5. **Computational Resources**: It took huge amounts of time to process data for all 369 volumes. We at times had to work with few amounts of data inorder to quickly fid bugs and do the code. In the end it still took alot of time nnot just to extract features but to split train on stratified k fold with validation all the while searching for best model through Grid Search.



