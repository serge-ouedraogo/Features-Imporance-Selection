# Breast Cancer Important Features
For this project I used a linear regression model to identify the most important features indicative of breast cancer. An exhaustive Exploratory Data Analysis was conducted identify and remove duplicate entries  and outliers. There are some missing values  
Data segmentation is a technique used to divide a population into groups based on shared behavior or demographics. Here the segmentation consists of an 
unsupervised learning algorithm that is used to divide customers for a credit card company into specifics group and to extract insights members 
of each group share in common.

The main steps include: 
#### I Data Inspection
  Missing values
````
def missing_value(df):
  col = df.columns
  for i in col:
    if df[i].isnull().sum()>0:
      df[i].fillna(df[i].mode()[0],inplace=True)
  `````
  #### II Data Exploration
  1. Identification of correlated features
  
  2. Detection and Removal of Outliers
  ````
def outlier(df,columns):
    for i in columns:
        quartile_1,quartile_3 = np.percentile(df[i],[25,75])
        quartile_f,quartile_l = np.percentile(df[i],[1,99])
        IQR = quartile_3-quartile_1
        lower_bound = quartile_1 - (1.5*IQR)
        upper_bound = quartile_3 + (1.5*IQR)
        ...

  ````
#### III Data Preprocessing 
  1. Feature Selection: Removing features with low variance
  ```
  def variance_threshold_selector(df, col):
    data = df[col].values
    X = data[:, 1:-1]
    ...
  ```
  2. Feature Transformation to uniformly rescale the dataset 
  3. Feature Reduction using PCA
  ````
  def pca_results(good_data, pca, col):
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data[col].keys()))
    components.index = dimensions
    
    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    ...
   ````
   
   #### IV Data Clustering using Gaussian Mixture Model
  ````
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

scores = {}
for i in range(2,11):
    
    print('Number of clusters: ' + str(i))
        
    # Apply your clustering algorithm of choice to the reduced data 
    clusterer = GaussianMixture(random_state=42, n_components=i)
    clusterer.fit(X_tf)

    # Predict the cluster for each data point
    preds = clusterer.predict(X_tf)
    ....
'''

