# ML-ProblemSolving-FrameWork (Note: in Progress)
<hr>

<center><h4><font color='blue'>Framework to convert Business Problem into Data Problem</font></h4></center>


Framework which I used:
1. `Understand Business problem` using `TOSCAR` framework
    * `T` : Trouble
    * `O` : Owner
    * `S` : Success Criteria
    * `C` : Constraints
    * `A` : Actors
    * `R` : Refrences

|T - Trouble |O - Owner |S - Success Criteria |C - Constraints |A - Actors |R - Refrences|
|------------|----------|---------------------|----------------|-----------|-------------|
|Why? |Who will own implementation? |How we will measure sucess? |How frequently a customer can be contacted in a day/week/month? |CEO (in our case) |Have we tried this in past? |
|Why now? |CEO (in our case) |Average spending of existing customer? |How much we can spend? |Opeartion team maybe next |What were results? |
|Why existing customers? |Sales Director maybe next |Percentage of customer buying over fixed time period? |Do we have any CRM? | Customer management|How current cross-sell or up-sell are being handled? |
|Any competition benchmark?|Customer management director maybe next |What to achive in which time frame? |How much past data do we have? |Sales Team | |
|Do we have any Cross-sell going on? |What does CEO feels about this problem? |Can we achive same sucess by other ways? |Do we capture behavioral data? |Director | |
|Where does it fit in your(CEO priority)?  |Do other in organization feals even this is a problem? ||Any regulation to follow or keep in head?  | | |


2. `Frame Problem Statement`
3. `Break points 2 into smaller problem`
4. `Convert Smaller problems to data problems`

---

* Classification Problem
* Regression Problem

---

## Getting Started

When given a new ML project, workflow could be as following:

1. **`Define Problem`** <br />
  Investigate and characterize the problem and clarify the project goal.

2. **`Summarize Data`** <br />
  Use descriptive statistics and visualization techniques to get a grasp of data. 
   - `Descriptive Statistics` <br />
     data dimension, type, attribute features (count, mean, std, min/max, percentiles), class categories, correlations between attributes, skew of univariate distributions
     
   - `Visualization` <br />
     univariate plots(histograms, density plot, boxplot), multivariate plots(correlation matrix plot, scatter plot matrix)

3. **`Data Preprocessing` [Incompleted]**
   
   3.1. **`Transformation`** <br />
       The reason for preprocessing data is that different algorithms make different assumptions about data requiring different transformation. Here are some common processing techniques:
     
     - *`Rescaling`* <br />
       To limit varying attributes ranges all between 0 and 1. Useful for weight-inputs regression/neural networks and kNN.
          
     - *`Standardization`* <br />
       To transform attributes with a Gaussian distribution to a standard Gaussian distribution (0 mean and 1 std). Useful for linear/logistic regression and LDA
     
     - *`Normalization`* <br />
       To rescaling each observation (row) to have a length of 1 (called a unit norm or a vector with the length of 1 in linear algebra). Useful for sparse dataset with varying attribute scales, weight-input neural network and kNN
   
     - *`Binarization`* <br />
       To transform data using a binary threshold.(1 for above threshold and 0 for below threshold)
     
   3.2. **`Feature Selection`** <br />
     Irrelevant or partially relevant features can negatively impact model performance, such as decreasing the accuracy of many models. Feature Selection is to select features that contribute most to the prediction variable or output in which you are interested. It can help reduce overfiting, improve accuracy, reduce training time. Here are some common processing techniques:
     - *`Statistical Test Selection with *chi-2*`* <br />
       To select those features that have the strongest relationship with output variable 
       
     - *`Recursive Feature Elimination (RFE)`* <br />  
       To recursively removing attributes and building a model on those attributes that remain.
       
     - *`Principal Component Analysis (PCA)`* <br />
       A kind of data reduction technique. It uses linear algebra to transform the dataset into a compressed form and choose the number of dimensions or principal components in the transformed result.
     
     - *`Feature importance`* <br />
     To use bagged decision trees such as Random Forest and Extra Trees to estimate the importance of features.

4. **`Algorithm Evaluation`**
   - `Separate train/test dataset (Resampling)` <br />
     In most cases, *k-fold* Cross Validation technique (e.g. k = 3, 5 or 10) will be used to estimate algorithm performance with less variance. At first, the dataset will be splited into *k* parts. Then the algorithm is trained on *k-1* folds with one held back and tested on the held back fold. Repeatedly, each fold of the dataset will be given a chance to be the held back test set. After all these, you can summarize using the mean and std of such *k* different performance scores.
     
   - `Performance Metrics` <br />
     Choice of metrics influence how the performance of ML algorithms is measure and compared, as it represents how you weight the importance of different characteristics in the output results and ultimate choice of which algorithm to choose.
     - **`For Classification Problem`**
       1) `Classification Accuracy`  <br />
          Classification Accuracy is the ratio of the number of correct predictions and the numberof all predictions. Only suitable for equal number of obsevations in each class, and all predictions and prediction errors are *equally important*.
       
       2) `Logorithmic Loss`  <br />
          Logorithmic Loss is to evaluate the predictions of probabilities of membership to a given class. Corrected or incorrected prediction errors will be rewarded/punished proportionally to the comfidence of the prediction. Smaller logloss is better with 0 for a perfect logloss.
       
       3) `Area Under ROC Curve (AUC)` <br />
          AUC is used to evaluate binary classification problem, representing a model's ability to discriminate between positive and negative classes. (1 for perfect prediction. 0.5 for as good as random.)  <br />
          ROC can be broken down into sensitivity (true positive rate) and specificity (true negative rate)
          
       4) `Confusion Matrix` <br />
          Confusion Matrix is representation of models' classes accuracy. Generally, the majority of the predictions fall on the diagonal line of the matrix.
          
       5) `Classification Report` <br />
          A scikit-learn lib provided classification report including precision, recall and F1-score.
          
     - **`For Regression Problem`**
       1) `Mean Absolute Error (MAE)` `L1-norm` <br />
          MAE is the sum of the absolute differences between predictions and actual values.
       
       2) `Mean Squared Error (MSE)` `L2-norm` <br />
          MSE is the sum of square root of the mean squared error
          
       3) `R^2`<br />
          R^2 is an indication of the goodness of fit of a set of predictions to the actual values range between 0 (non-fit) and 1 (perfiect fit). Statistically, it is called coefficient of determination
          
       
 
    - `Spot-Checking Algorithm` <br />
      Spot-Checking is a way to discover which algs perform well on ML problem. Since we do not know which algs will work best on our dataset, we can make a guess and further dig deeper. Here are some common algs:
      - `Classification`
        - Logistic Regression (Linear)
        - Linear Discriminant Analysis (Linear)
        - k-Nearest Neighbors (Non-linear)
        - Naive Bayes (Non-linear)
        - Classification and Regression (Non-linear)
        - Support Vector Machine (Non-linear)
        
      - `Regression`
        - Linear Regression (Linear)
        - Ridge Regression (Linear)
        - LASSO Linear Regression (Linear)
        - Elastic Net Regression (Linear)
        - Naive Bayes (Non-linear)
        - Classification and Regression (Non-linear)
        - Support Vector Machine (Non-linear)

5. **`Improve Results`**
   - `Ensemble` <br />
     Ensemble learning helps improve machine learning results by combining several models. This approach allows the production of better predictive performance compared to a single model.
     
     1) `Bagging` <br />
        Bagging tries to implement similar learners on small sample populations and then takes a mean of all the predictions.
        
     2) `Boosting` <br/>
        Boosting is an iterative technique which adjust the weight of an observation based on the last classification. If an observation was classified incorrectly, it tries to increase the weight of this observation and vice versa. Boosting in general decreases the bias error and builds strong predictive models. However, they may sometimes over fit on the training data.
        
     3) `Stacking`<br />
        Here we use a learner to combine output from different learners. This can lead to decrease in either bias or variance error depending on the combining learner we use.
   
   - `Params Tuning` <br />
     Algs tuning is the final step in AML before finalizing model. In scikit-learn, there are two simple methods for params tuning:
     
     1) `Grid Search Param Tuning` <br />
        Methodically build and evaluate a model for each combination of algorithm parameters specified in a grid
        
     2) `Random Search Param Tuning` <br />
        Sample algorithm parameters from a random distribution (i.e. uniform) for a fixed number of iterations. A model is constructed and evaluated for each combination of parameters chosen.       


Using hyper-parameter optimization, we can also find correct penalty to use

|Model                |  Optimize                       |  Range of values             |
|---------------------|---------------------------------|------------------------------|
|`Linear Regression`  |- fit_intercept <br> - normalize |- True/False <br> - True/False|
|`Logistic Regression`|- Penalty <br> - C               |- l1 or l2 <br> - 0.001, 0.01.....10...100|
|`k-NN`               |- n_neighbors <br> -p            |- 2, 4, 8, 16... <br> - 2,3   |
|`SVM`                |-C <br> - gamma <br> - class_weight|- 0.001,0.01..10..100..1000 <br> - scale, auto or float, RS* <br> - balanced , None|
| `Ridge`             |- alpha <br> - fit_intercept <br> - normalize |- 0.01, 0.1, 1.0, 10, 100 <br> - True/False <br> - True/False|
|`Lasso`              |- alpha <br> - normalize                      |- 0.1, 1.0, 10 <br> - True/False|
|`Decision Tree`      |- criterion <br> - splitter <br> - max_depth  |- gini, entropy  <br> -best, random <br> - 5,8,10|
|`Random Forest`      |- n_estimators <br> - max_depth <br> - min_samples_split <br> - min_samples_leaf <br> - max features|- 120, 300, 500, 800, 1200 <br> - 5, 8, 15, 25, 30, None <br> - 1,2,5,10,15,100  <br> - 1,2,5,10 <br>- log2, sqrt, None|
|`XGBoost`            |- eta <br> - gamma <br> - max_depth <br> - min_child_weight <br> - subsample <br> - colsample_bytree <br> - lambda <br> - alpha| - 0.01,0.015, 0.025, 0.05, 0.1 <br> - 0.05-0.1,0.3,0.5,0.7,0.9,1.0 <br> - 3,5,7,9,12,15,17,25 <br> - 1,3,5,7 <br> - 0.6, 0.7, 0.8, 0.9, 1.0 <br> - 0.6, 0.7, 0.8, 0.9, 1.0 <br> - 0.01-0.1, 1.0 , RS* <br> - 0, 0.1, 0.5, 1.0 RS*|
               
             
6. **`Show Results`** <br />
   - Predictions on validation dataset
   - Create standalone model on entire training dataset
   - Save model for later use

#### [Refer This For More](https://www.kaggle.com/code/mukeshmanral/machine-learning-life-cycle-basic/notebook)
