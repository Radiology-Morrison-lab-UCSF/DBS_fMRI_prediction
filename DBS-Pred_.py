import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from math import log10
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error, make_scorer, roc_curve, roc_auc_score
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from statsmodels.api import OLS
from nilearn import datasets
from PIL import Image
from joblib import Parallel, delayed
from collections import defaultdict
from sklearn.manifold import TSNE



#############################
#         DATA LOADING      #
#############################

# Load main data file with clinical info
dataJAN_fin = pd.read_excel('dataJAN_fin.xlsx', index_col=0)

# Load functional data
data_fc = loadmat('allrois_fc_mar.mat')
data2 = loadmat('allrois_fc_ctrl.mat')
dataht = loadmat('allroisfc_ht.mat')

func = data_fc['allrois_fc']
funcctrl = data2['allrois_fc']
funcht = dataht['allrois_fc']

#%%
frst = []
scnd = []
thrd = []
fort = []

#############################
#     PHENOTYPIC EXTRACTION #
#############################

post1_raw = np.array(dat3_cl['LEDDpost1_Delta'])
post2_raw = np.array(dat3_cl['LEDDpost2_Delta'])
post3_raw = np.array(dat3_cl['LEDDpost3_Delta'])
post4_raw = np.array(dat3_cl['LEDDpost4_Delta'])

# Extract improvement (LEDD delta) by taking first available post-op measure
improvs = []
nan_indices = [108] #data missing
for i in range(len(post1_raw)):
    if np.isnan(post1_raw[i]) and np.isnan(post2_raw[i]) and np.isnan(post3_raw[i]) and np.isnan(post4_raw[i]):
        nan_indices.append(i)
    else:
        if not np.isnan(post1_raw[i]) and not i == 108:
            frst.append(i)
            improvs.append(post1_raw[i])
        elif not np.isnan(post2_raw[i]) and not i == 108:
            scnd.append(i)
            improvs.append(post2_raw[i])
        elif not np.isnan(post3_raw[i]) and not i == 108:
            thrd.append(i)
            improvs.append(post3_raw[i])
        elif not np.isnan(post4_raw[i]) and not i == 108:
            fort.append(i)
            improvs.append(post4_raw[i])

improvs = np.array(improvs).astype(float)


# Filter functional data to match subjects without NaNs
funcready = np.delete(func, nan_indices, axis=0)

#############################
#  UPPER-TRIANGULAR EXTRACT #
#############################
upper_triangular_indices = np.triu_indices(36, k=1)
rows4, cols4 = upper_triangular_indices
upper_triangular_values = np.zeros((len(funcready), len(rows4)))

# Extract upper triangular values from each subject's connectivity matrix
for i in range(len(funcready)):
    upper_triangular_values[i] = funcready[i][upper_triangular_indices]

funcreadytr = upper_triangular_values


mds = np.array(dat3_cl['MDSUPDRSIIIpre_Percent_TOTAL_V'])
mds = np.delete(mds, nan_indices)
mds = mds.astype(float)
mds = mds.reshape(len(mds),1) 
#Check for NaN and interpolate
if np.where(np.isnan(mds))[0] is not None:
    naninds = np.where(np.isnan(mds))[0]
    print(naninds)
    mds[naninds] = np.nanmean(mds)
mds = mds.reshape(len(mds),1) 

dp = improvs  # LEDD change
indp = funcreadytr
indpctrl = funcctrl
indpht = funcht.reshape(np.shape(funcht)[0], -1)


#############################
#   EXTRACT COVARIATES      #
#############################

#%% Map Target regions - This step does NOT need to be done. The provided .xlsx 
# is already preprocessed for this step
target_map = {
    'STN': 0, 
    'GPi': 1, 
    'GPi ': 1, 
    'STN &\nSubgaleal': 0, 
    'STN & Subgaleal': 0, 
    'STN + Subgaleal': 0,  
    'STN/GPi?': 0.5, 
    'STN/GPi': 0.5, 
    'VIM': 2
}
dataJAN_fin['Target_L_R'] = dataJAN_fin['Target_L_R'].replace(target_map)
dataJAN_fin['Race'] = dataJAN_fin['Race'].replace({'white': 0, 'white ': 0, 'native american or alaska native/ white': 0, 'hispanic': 1, 'asian': 2, 'black': 3, 'black or african american': 3})
dataJAN_fin['Sex'] = dataJAN_fin['Sex'].replace({'M': 0, 'F': 1})
#%%

yearspd = np.array(dataJAN_fin['Years_PD_at_SX_MRI'])
gender = np.array(dataJAN_fin['Sex'])
age = np.array(dataJAN_fin['age'])
target = np.array(dataJAN_fin['Target_L_R'])
billist = np.array(dataJAN_fin["No_Leads"].replace({'Bi': 0, 'BI ': 0, 'BI': 0, 'Bi ': 0, 'L': 1, 'R' : 1, 'Bi STN\nL GPi\n(3 total)' : 0, "Bi (4 total)" : 0 }))
retrclin = np.array(dataJAN_fin["PTID_Retro_Clin"])
race = np.array(dataJAN_fin['Race'])
ids = np.array(dataJAN_fin["PTID_Retro_Clin"])
#%%
# Remove NaNs
gender = np.delete(gender, nan_indices, axis=0)
age = np.delete(age, nan_indices)
target = np.delete(target, nan_indices)
yearspd = np.delete(yearspd, nan_indices)
billist = np.delete(billist, nan_indices)
retrclin = np.delete(retrclin, nan_indices)
race = np.delete(race, nan_indices)
ids = np.delete(ids, nan_indices)


# Reshape for modeling
genderr = gender.reshape(-1,1)
yearspdd = yearspd.reshape(-1,1)
agee = age.reshape(-1,1)
targett = target.reshape(-1,1)
racee = race.reshape(-1,1)

# Index 102 has missing data
z_func = np.delete(indp, 102, axis=0)
dp = np.delete(dp, 102, axis=0)
genderr = np.delete(genderr, 102, axis=0)
agee = np.delete(agee, 102, axis=0)
yearspdd = np.delete(yearspdd, 102, axis=0)
targett = np.delete(targett, 102, axis=0)
mds = np.delete(mds, 102, axis=0)
billist = np.delete(billist, 102, axis=0)
racee = np.delete(racee, 102, axis=0)
ids = np.delete(ids, 102, axis=0)

funcready = np.delete(funcready, 102, axis=0)

#%% Pt removed their consent
z_func = np.delete(z_func, -3, axis=0)
dp = np.delete(dp, -3, axis=0)
genderr = np.delete(genderr, -3, axis=0)
agee = np.delete(agee, -3, axis=0)
yearspdd = np.delete(yearspdd, -3, axis=0)
targett = np.delete(targett, -3, axis=0)
mds = np.delete(mds, -3, axis=0)
billist = np.delete(billist, -3, axis=0)
funcready = np.delete(funcready, -3, axis=0)
racee = np.delete(racee, -3, axis=0)
ids = np.delete(ids, -3, axis=0)

#%% VIM pt removed
z_func = np.delete(z_func, 65, axis=0)
dp = np.delete(dp, 65, axis=0)
genderr = np.delete(genderr, -3, axis=0)
agee = np.delete(agee, 65, axis=0)
yearspdd = np.delete(yearspdd, 65, axis=0)
targett = np.delete(targett, 65, axis=0)
mds = np.delete(mds, 65, axis=0)
billist = np.delete(billist, 65, axis=0)
funcready = np.delete(funcready, 65, axis=0)
racee = np.delete(racee, 65, axis=0)
ids = np.delete(ids, 65, axis=0)

#%%
#Labels for classification - PD vs HC
ones_array = [1] * 120
zeros_array = [0] * 75
labels = ones_array + zeros_array

# PD vs HC2
ones_array = [1] * 120
zeros_array = [0] * 32
labels1 = ones_array + zeros_array

#HD vs HC
ones_array2 = [1] * 93
zeros_array = [0] * 75
labels2 = ones_array2 + zeros_array

#PD vs HD
ones_array2 = [1] * 120
zeros_array = [0] * 93
labels3 = ones_array2 + zeros_array

#HD vs HC2
ones_array2 = [1] * 93
zeros_array = [0] * 32
labels4 = ones_array2 + zeros_array

#############################
#   REGION NAMES (36 Atlas) #
#############################

names = [
    "Globus-Pallidus-E l","Globus-Pallidus-E r",
    "Globus-Pallidus-I l","Globus-Pallidus-I r",
    "Ventral-Pallidum l","Ventral-Pallidum r",
    "Substantia-Nigra-pr r","Substantia-Nigra-pr l",
    "Substantia-Nigra-pc r","Substantia-Nigra-pc l",
    "TL-Anterior r", "TL-Anterior l",
    "Caudate-Nucleus r","Caudate-Nucleus l",
    "Putamen r","Putamen l",
    "Cerebelum-3 l","Cerebelum-3 r",
    "Cerebelum-45 l","Cerebelum-45 r",
    "Cerebelum-6 l","Cerebelum-6 r",
    "Dentate-Nucleus r","Dentate-Nucleus l",
    "Postcentral l","Postcentral r",
    "Precentral l","Precentral r",
    "Supp-Motor-Area l","Supp-Motor-Area r",
    "Subthalamic-Nucleus l","Subthalamic-Nucleus r",
    "Red-Nucleus r","Red-Nucleus l",
    "Nuc-Accumbens l","Nuc-Accumbens r"
]

#Regions for the earlier 60-ROI version
names_old = ["Globus-Pallidus-E l",
"Globus-Pallidus-E r",
"Globus-Pallidus-I l",
"Globus-Pallidus-I r",
"Ventral-Pallidum l",
"Ventral-Pallidum r",
"Substantia-Nigra-pr r",
"Substantia-Nigra-pr l",
"Substantia-Nigra-pc r",
"Substantia-Nigra-pc l",
"TL-Anterior r",
"TL-Anterior l",
"TL-Medial-nucleus r",
"TL-Medial-nucleus l",
"TL-Midine-thalamic-nulcei r",
"TL-Midine-thalamic-nulcei l",
"TL-Pulvinar r",
"TL-Pulvinar l",
"TL-Internal-Medullary_Lamina r",
"TL-Internal-Medullary_Lamina l",
"TL-Lateral-Nucleus r",
"TL-Lateral-Nucleus l",
"Caudate-Nucleus r",
"Caudate-Nucleus l",
"Putamen r",
"Putamen l",
"Cerebelum-Crus1 l",
"Cerebelum-Crus1 r",
"Cerebelum-Crus2 l",
"Cerebelum-Crus2 r",
"Cerebelum-3 l",
"Cerebelum-3 r",
"Cerebelum-45 l",
"Cerebelum-45 r",
"Cerebelum-6 l",
"Cerebelum-6 r",
"Cerebelum-7b l",
"Cerebelum-7b r",
"Cerebelum-8 l",
"Cerebelum-8 r",
"Cerebelum-9 l",
"Cerebelum-9 r",
"Cerebelum-10 l",
"Cerebelum-10 r",
"Dentate-Nucleus r",
"Dentate-Nucleus l",
"Postcentral l",
"Postcentral r",
"Precentral l",
"Precentral r",
"Supp-Motor-Area l",
"Supp-Motor-Area r",
"Subthalamic-Nucleus l",
"Subthalamic-Nucleus r",
"Red-Nucleus r",
"Red-Nucleus l",
"Nuc-Accumbens l",
"Nuc-Accumbens r",
"Hippocampus r",
"Hippocampus l"]

#############################
#     DATA NORMALIZATION    #
#############################

scaler_X = StandardScaler()
z_func_s = scaler_X.fit_transform(z_func)

dp = dp.reshape(-1,1)
scaler_y = StandardScaler()
dp_s = scaler_y.fit_transform(dp)

# Combine functional data with clinical covariates
withcov = np.concatenate((z_func_s, mds, agee, genderr, targett), axis=1).astype(float)


#############################
#       HELPER FUNCTIONS    #
#############################

def map_to_original(index, rows=rows4, cols=cols4):
    """
    Maps a linear index back to the original (row, column) pair in the connectivity matrix.

    Parameters
    ----------
    index : int
        Linear index.
    rows : array-like
        Row indices of upper triangular connectivity elements.
    cols : array-like
        Column indices of upper triangular connectivity elements.

    Returns
    -------
    tuple
        (row, column) indices in the original matrix.
    """
    return rows[index], cols[index]


def fisher_transform(r):
    """
    Performs Fisher's z-transformation on a correlation coefficient.

    Parameters
    ----------
    r : float
        Correlation coefficient.

    Returns
    -------
    float
        Fisher z-transformed value.
    """
    return 0.5 * np.log((1 + r) / (1 - r))


def feats_uniq(selected_pairs, funcy):
    """
    Extracts features (connectivity values) from a set of selected pairs of regions.

    Parameters
    ----------
    selected_pairs : list
        List of (row, column) pairs indicating which ROI-ROI connections to extract.
    funcy : ndarray
        Functional connectivity data (subjects x ROIs x ROIs).

    Returns
    -------
    allun : np.ndarray
        Unique sorted indices of selected connections.
    feats : np.ndarray
        Extracted features (subjects x number_of_selected_features).
    """
    alll = np.array(selected_pairs)
    # Get unique ROI indices
    allun = np.unique(alll, axis=0)

    rowy = allun[:,0]
    coly = allun[:,1]
    feats = funcy[:,rowy,coly]

    return allun, feats

#%%
def lassoreg_coeffs(indp, dp, alpha):
    
    '''    
    Parameters
    ----------
    indp : numpy array (float) 
        The independent variable (connectivity matrix & phenotypic information)
    dp : numpy array (float)
        The dependent variable (delta LEDD scores)
    alpha : float
        Regularization parameter

    Returns
    -------
    mostyy : list
        Indices for the selected rows and columns from the corr matrix
    sig_cor : list
        Names of the selected functional pairs
    coefficients : numpy array
        Model-assigned coefficients for the predictors
    '''
    conv = 1
        # Create a Lasso Regression model
    lasso_model = Lasso(alpha=alpha)  # Set the regularization parameter alpha
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        lasso_model.fit(indp, dp)
        
        # Check for convergence warnings
        if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
            conv=0
    
    # Fit the Lasso model to the data
    # Get the coefficients of the Lasso model
    coefficients = lasso_model.coef_
    store = []
    coefsstore = []
    # Sorts the coefficients and extracts the top 10
    sorty = np.argsort(np.abs(coefficients))[::-1]
    for i in range(len(sorty)):
        if coefficients[sorty[i]] != 0 and sorty[i] < len(z_func_s[0]):  
            store.append(sorty[i])
            coefsstore.append(coefficients[sorty[i]])
   
    sorty2 = []
    for i in range(len(sorty)):
        if sorty[i] < len(z_func_s[0]):
            sorty2.append(sorty[i])
    mosty = np.array(sorty2[:10])
    # Gets the features in an interpretable form to the output variables
    origshape = (36, 36)
    rows = mosty // origshape[1]
    cols = mosty % origshape[1]
    mostyy = []
    for i in range(len(mosty)):      
      mostyy.append([map_to_original(mosty[i])[0], map_to_original(mosty[i])[1]])
    
    store = np.array(store)
    rows = store // origshape[1]
    cols = store % origshape[1]
    sortyy = []
    for i in range(len(store)):      
      sortyy.append([map_to_original(store[i])[0], map_to_original(store[i])[1]])
      
      
    sig_cor = []
    for elem in mostyy:
      sig_cor.append(names[elem[0]] + ' and ' + names[elem[1]])
    return mosty, mostyy, sig_cor, coefficients, conv, sortyy, coefsstore

#%%

def svm_coeffs(indp, dp):
    num_subjects = indp.shape[0]
    num_cells = indp.shape[1]
    # Prepare the data for SVM regression
    X = indp  # Features
    y = dp  # Target variable
    # Create and fit the SVM regression model
    svm = SVR(kernel='linear')
    svm.fit(X, y)
    # Get the coefficients of the SVM model
    svm_coefs = np.abs(svm.coef_).flatten()

    # Find the indices of the cells with significant correlation
    correlated_indices = svm_coefs.argsort()[::-1]
    origshape = (36, 36)
    rowsss = correlated_indices // origshape[1]
    colsss = correlated_indices % origshape[1]
    featuresss = []
    for i in range(len(correlated_indices)):
      featuresss.append([rowsss[i], colsss[i]])
    featuresss = featuresss[:10]
    sig_cor = []
    for elem in featuresss:
      sig_cor.append(names[elem[0]] + ' and ' + names[elem[1]])
    return featuresss, sig_cor

#%%

def rfe_linreg(indp, dp):
  # Create a base model (e.g., linear regression)
  base_model = LinearRegression()

  # Create an RFE object with the base model and specify the desired number of features to select
  rfe = RFE(estimator=base_model, n_features_to_select=25)

  # Fit the RFE object to the data
  rfe.fit(indp, dp)

  # Get the selected features (mask of selected features)
  selected_features_mask = rfe.support_ 

  # Get the rank of each feature (ranking of features, with a higher value indicating higher importance)
  feature_ranks = rfe.ranking_

  # Print the selected features and their ranks
  features = []
  for feature, rank in zip(range(indp.shape[1]), feature_ranks):
      if selected_features_mask[feature]:
          features.append(feature+1)
  features = np.array(features)
  origshape = (36, 36)
  rows = features // origshape[1]
  cols = features % origshape[1]
  mostyy = []
  for i in range(len(features)):
    mostyy.append([rows[i], cols[i]])

  mostyy = mostyy[:10]
  sig_cor = []
  for elem in mostyy:
    sig_cor.append(names[elem[0]] + ' and ' + names[elem[1]])
  return mostyy, sig_cor

#%%
def randfors(indp, dp):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(indp, indp, test_size=0.2, random_state=0)

    # Initialize the model for regression
    rf = RandomForestRegressor(n_estimators=100, random_state=0)

    # Train the model
    rf.fit(X_train, y_train)


    # Feature selection
    feature_importances = rf.feature_importances_
    threshold = np.mean(feature_importances) * 2  # For example, twice the mean importance
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    ally = np.array(list(range(634)))
    selected_features = ally[list(selector.get_support())]

    origshape = (36, 36)
    rowsss = selected_features // origshape[1]
    colsss = selected_features % origshape[1]
    featuresss = []
    for i in range(len(selected_features)):
        featuresss.append([rowsss[i], colsss[i]])

    randfor = featuresss[:10]
    return randfor


#%%
def find_frequent_pairs(input_dict):

    """
    Finds frequent pairs of features that appear in more than 50% of the entries in input_dict.

    Parameters
    ----------
    input_dict : dict
        Keys are iteration indices, values are lists of selected feature pairs.

    Returns
    -------
    frequent_pairs : list
        List of pairs that occur more than half of the time.
    """
    pair_count = defaultdict(int)
    total_lists = 0  # Count how many valid lists we have
    
    # Count occurrences of each pair
    for key, value in input_dict.items():
        if isinstance(value, list):
            total_lists += 1
            for pair in value:
                if isinstance(pair, list) and len(pair) == 2:
                    pair_count[tuple(pair)] += 1
    
    # Calculate threshold for >75%
    threshold = total_lists / 1.5
    frequent_pairs = []
    print(pair_count)
    for pair, count in pair_count.items():
        if count > threshold:
            frequent_pairs.append(list(pair))
    
    return frequent_pairs


# Example dictionary to find frequent pairs
#example_dict = lasso
#result = find_frequent_pairs(example_dict)
#print(len(result))

#%%
def test_r2_tott(indp, dp):
    """
    Fits an OLS model and returns regression diagnostics.

    Parameters
    ----------
    indp : np.ndarray
        Independent variables.
    dp : np.ndarray
        Dependent variable.

    Returns
    -------
    significant_indices : tuple
        Indices of features with p < 0.05.
    sorted_p_values : np.ndarray
        Sorted p-values of significant features.
    predictions : np.ndarray
        Model predictions.
    residuals : np.ndarray
        Residuals of the model.
    aic : float
        Akaike Information Criterion.
    adjusted_r_squared : float
        Adjusted R-squared of the model.
    summary : pd.Series
        Model parameters.
    """
    model = OLS(dp, indp).fit()
    predictions = model.predict(indp)
    residuals = dp - predictions
    p_values = model.pvalues
    aic = model.aic
    adjusted_r_squared = model.rsquared_adj
    summary = model.params
    significant_indices = np.where(p_values < 0.05)
    sorted_p_values = p_values[significant_indices]
    return significant_indices, sorted_p_values, predictions, residuals, aic, adjusted_r_squared, summary, p_values
#%%

def generate_indices(num_sets, set_size, max_value):
    indices_list = []
    for _ in range(num_sets):
        # Generate a random set of indices from 1 to max_value (inclusive)
        indices = np.random.choice(range(max_value), set_size, replace=False)
        indices_list.append(indices)
    
    return indices_list
#%%

def test(alpha, indp, dp, fc, comb=1, alll=1):
    '''
    
    Parameters
    ----------
    alpha : float
        Lasso penalty.
    comb : binary
        0 when I want to only use clinical features 
    alll : binary
        0 when I want to only run Lasso

    Returns
    -------
    lasso : list
        Selected features (indices of connections in the conn. matrix)
    rsq_las : list
        Adj r-squared using Lasso features
    rsq_svm : list
        Adj r-squared using SVM features
    rsq_rfe : list
        Adj r-squared using Lasso features
    rsq_rfs : list
        Adj r-squared using Lasso features
    unconverged_count : list
        Runs that didn't converge

    '''
    # Generate 501 different sets of indices from the range 1 to 500
    num_sets = 10000
    set_size = 65
    max_value = 80
    index_sets = generate_indices(num_sets, set_size, max_value)
    
    # Print the first few sets to verify
    for i, indices in enumerate(index_sets[:5]):
        print(f"Set {i+1}: {indices}")
    lasso = {}
    svm = []
    rfe = []
    randfor = []
    rsq_las = [] #I edit this when I try different alpha values or combination of predictors
    rsq_svm = []
    rsq_rfe = []
    rsq_rfs = []
    coeffy = {}
    rms = []
    unconverged_count = [] #To track model convergence for different alpha values
    i = 0
    test_indices = index_sets[5000:];
    
    for elem, test_index in zip(index_sets[:5000], test_indices):
        i += 1
        print(f"Iteration: {i}")  # To check code running speed
       
        # Split data into training and testing sets
        x_train, _ = withcov[elem], withcov[test_index]
        y_train, y_test = dp_s[elem], dp_s[test_index]
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
    
        
        age_train, age_test = agee[elem], agee[test_index]
        gender_train, gender_test = genderr[elem], genderr[test_index]
        target_train, target_test = targett[elem], targett[test_index]
    
        # Lasso Regression
        _, lassoreg, _, _, conv,lasstore, coefsstore = lassoreg_coeffs(x_train, y_train, alpha) 
        unconverged_count.append(conv)
        lasso[i] = lasstore
        coeffy[i] = coefsstore
        # Feature selection for Lasso
        _, featslas = feats_uniq(lassoreg, funcready[test_index,:,:])
        if comb == 1:
            indp_fl_las = np.concatenate((featslas, mds[test_index], age_test, gender_test, target_test), axis=1).astype(float)
        else:
            indp_fl_las = np.concatenate((mds[test_index], age_test, gender_test, target_test), axis=1).astype(float)
        
        # Standardization of the feature data
        scaler_X = StandardScaler()
        indp_fl_las_s = scaler_X.fit_transform(indp_fl_las)
        

        
        # Get adjusted R^2 for Lasso and SVM
        _, _, _, _, _, adjlas, summary, _ = test_r2_tott(indp_fl_las_s, y_test)
        rsq_las.append(adjlas)



        if alll == 1:
            # SVM Regression
            svmreg, _ = svm_coeffs(x_train, y_train)
            svm.append(svmreg)
            
            #RFE
            rfereg, _ = rfe_linreg(x_train, y_train)
            rfe.append(rfereg)
            
            #RandFor
            randforreg = randfors(x_train, y_train)
            randfor.append(randforreg) 
             
            # Feature selection for SVM
            _, featssvm = feats_uniq(svmreg, funcready[test_index,:,:])
            indp_fl_svm = np.concatenate((featssvm, age_test, gender_test, target_test), axis=1).astype(float)
            scaler_X = StandardScaler()
            indp_fl_svm_s = scaler_X.fit_transform(indp_fl_svm)
    
            # Feature selection for RFE
            _,featsrfe = feats_uniq(rfereg, funcready[test_index,:,:])
            indp_fl_rfe = np.concatenate((featsrfe, age_test, gender_test, target_test), axis=1) 
            indp_fl_rfe = indp_fl_rfe.astype(float)
            scaler_X = StandardScaler()
            indp_fl_rfe_s = scaler_X.fit_transform(indp_fl_rfe)
            
            # Feature selection for RandFor
            _,featsrfs = feats_uniq(randforreg, funcready[test_index,:,:])
            indp_fl_rfs = np.concatenate((featsrfs, age_test, gender_test, target_test), axis=1) 
            indp_fl_rfs = indp_fl_rfs.astype(float)
            scaler_X = StandardScaler()
            indp_fl_rfs_s = scaler_X.fit_transform(indp_fl_rfs)
            
            _, _, _, _, _, adjsvm,_ = test_r2_tott(indp_fl_svm_s, y_test)
            _,_,_,_,_,adjrfe,_ = test_r2_tott(indp_fl_rfe_s, y_test)
            _,_,_,_,_,adjrfs,_ = test_r2_tott(indp_fl_rfs_s, y_test)
        
            rsq_svm.append(adjsvm)
            rsq_rfe.append(adjrfe) 
            rsq_rfs.append(adjrfs) 
        else:
            continue
    return lasso, rsq_las, rsq_svm, rsq_rfe, rsq_rfs, unconverged_count, lasstore, coeffy


#%% Run!

all_indices = np.arange(120)  # Assuming total subjects are 120
adj_r2_ho_tot_01 = []
result_alliter = []
siginds = []
pvals = []
coefs = []
for i in range(100):
    # Create 100 different held-out sets for validation
    hold_out_indices = np.random.choice(all_indices, size=40, replace=False)
    train_indices = np.setdiff1d(all_indices, hold_out_indices)
    
    # Create the held-out dataset
    withcov_ho = withcov[hold_out_indices]
    dp_s_ho = dp_s[hold_out_indices]
    funcready_ho = funcready[hold_out_indices]
    mds_ho = mds[hold_out_indices]
    agee_ho = agee[hold_out_indices]
    genderr_ho = genderr[hold_out_indices]
    targett_ho = targett[hold_out_indices]
    
    # Create the training dataset
    withcovv = withcov[train_indices]
    dp_ss = dp_s[train_indices]
    funcreadyy = funcready[train_indices]
    mdss = mds[train_indices]
    ageee = agee[train_indices]
    genderrr = genderr[train_indices]
    targettt = targett[train_indices]
    z_func_ss = z_func_s[train_indices]  # If z_func_s was defined, also limit it to training
    
    # Run the model on training data
    lasso, _, _, _, _, _, _, _= test(0.1, z_func_ss, dp_ss, funcreadyy, 1, 0)    
    # Get stable features over the 5000 bootstrap iterations for this train set
    result = find_frequent_pairs(lasso)
    result_alliter.append(result)
    print(len(result))
    _, feats_ho_full = feats_uniq(result, funcready_ho)
    
    # Define independent variable
    X_ho_final = np.concatenate((feats_ho_full, mds_ho, agee_ho, genderr_ho, targett_ho), axis=1).astype(float)
    scaler_X = StandardScaler()
    X_ho_final_s = scaler_X.fit_transform(X_ho_final)
    
    # Compute adjusted R² for this hold-out, which will average over 100 runs
    sigind, _, _, _, _, adj_r2_ho, summary, pval = test_r2_tott(X_ho_final_s, dp_s_ho)
    pvals.append(pval)
    siginds.append(sigind)
    coefs.append(summary)
    adj_r2_ho_tot_01.append(adj_r2_ho)
    

#%%
from collections import Counter

# Convert lists to tuples for hashing
pair_counts = Counter(tuple(pair) for sublist in result_alliter for pair in sublist)

# Define threshold (at least 25 out of 50 lists)
threshold = 75

# Get pairs that appear in at least 25 lists
frequent_pairs = [pair for pair, count in pair_counts.items() if count >= threshold]

# Display the result
frequent_pairs
#%% Get Final Feature Set

# Flatten the nested list of features
all_pairs = [tuple(pair) for sublist in result_alliter for pair in sublist]

# Count occurrences
pair_counts = Counter(all_pairs)

# Get the most common pairs for the final set
final_set = pair_counts.most_common()

#%%

# Updated and old ROI names for extraction
list_of_36 = names
list_of_60 = names_old

# Example list of pairs of indices corresponding to list_of_36
# 'result' contains pairs in terms of the 36-region atlas
pairs_in_36 = np.array(frequent_pairs).copy()

# Map from 36-region atlas to 60-region atlas
mapping_36_to_60 = {string: list_of_60.index(string) for string in list_of_36 if string in list_of_60}

# Convert each pair of indices from 36-region names to 60-region names
pairs_in_60 = [[mapping_36_to_60[list_of_36[i]], mapping_36_to_60[list_of_36[j]]] for i, j in pairs_in_36]

# Print original and new mappings
print("Original pairs (in list of 36):", pairs_in_36)
print("New pairs (in list of 60):", pairs_in_60)

#%% Extract significant correlations by name
sig_cor = []
sig_con = []
for i in range(len(frequent_pairs)):
    sig_cor.append(names[frequent_pairs[i][0]] + ' and ' + names[frequent_pairs[i][1]])
    sig_con.append([frequent_pairs[i][0], frequent_pairs[i][1]])

#%% Get features for PD, HC and HD groups
_, featslas = feats_uniq(sig_con, funcready)
_, feats2 = feats_uniq(pairs_in_60, funcctrl)
_, featsht = feats_uniq(pairs_in_60, funcht)

#%% Create combined datasets with PD, HC, and HD, for classification
indp_withctrl = np.concatenate((featslas, feats2), axis=0)
indp_withht = np.concatenate((featsht, feats2), axis=0)
indp_withboth = np.concatenate((featslas, featsht), axis=0)

#Standardize data
scaler_y = StandardScaler()
indp_withctrl_s = scaler_y.fit_transform(indp_withctrl)
scaler_z = StandardScaler()
indp_withht_s = scaler_z.fit_transform(indp_withht)
scaler_m = StandardScaler()
indp_withboth_s = scaler_m.fit_transform(indp_withboth)


#%% Plot ROC curves for classification tasks (PD vs HC, HD vs HC, PD vs HD)
# Using LogisticRegression and cross_val_predict

model = LogisticRegression()
y_scores = cross_val_predict(model, indp_withctrl, labels, cv=5, method="decision_function")

# Compute ROC curve and AUC for PD vs HC
fpr, tpr, thresholds = roc_curve(labels, y_scores)
auc_score = roc_auc_score(labels, y_scores)

model2 = LogisticRegression()
y_scores2 = cross_val_predict(model2, indp_withht, labels2, cv=5, method="decision_function")

# Compute ROC curve and AUC for HD vs HC
fpr2, tpr2, thresholds2 = roc_curve(labels2, y_scores2)
auc_score2 = roc_auc_score(labels2, y_scores2)

model3 = LogisticRegression()
y_scores3 = cross_val_predict(model3, indp_withboth, labels3, cv=5, method="decision_function")

# Compute ROC curve and AUC for PD vs HD
fpr3, tpr3, thresholds3 = roc_curve(labels3, y_scores3)
auc_score3 = roc_auc_score(labels3, y_scores3)

# Plotting the ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'PD vs HC - AUC = {auc_score:.2f}')
plt.plot(fpr2, tpr2, color='brown', lw=2, label=f'HD vs HC - AUC = {auc_score2:.2f}')
plt.plot(fpr3, tpr3, color='green', lw=2, label=f'PD vs HD - AUC = {auc_score3:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc="lower right", fontsize=20)
plt.grid(False)
plt.show()

#%%
# Combine features with 'correlations' array for the second HC group
indpwithnewctrlht = np.concatenate((featsht, correlations), axis=0)

#%% Histogram plot of a chosen feature to compare classes
labels_array = np.array(labels3)
feature_index = 0
feature_values = indp_withboth[:, feature_index]

plt.figure(figsize=(8, 6))
colors = ['green', 'blue']
class_names = ['HD', 'PD']
for class_value, color, class_name in zip([0, 1], colors, class_names):
    feature_values_class = feature_values[labels_array == class_value]
    print(f"Feature values in class '{class_name}' ({class_value}):", len(feature_values_class))
    plt.hist(
        feature_values_class, bins=20, alpha=0.7, color=color, label=class_name, density=True
    )
plt.xlabel('L Cereb 3 - L Cereb 4-5 FC', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=18)
plt.grid(False)
plt.show()

#%% Same but with two Histograms on same plot

labels_array = np.array(labels4)
feature_index = 0
feature_values = indpwithnewctrlht[:, feature_index]

plt.figure(figsize=(8, 6))

# Plot the original two classes: HC2 (class_value=0) and PD (class_value=1)
colors = ['brown', 'green']
class_names = ['HC2', 'HD']
for class_value, color, class_name in zip([0, 1], colors, class_names):
    feature_values_class = feature_values[labels_array == class_value]
    print(f"Feature values in class '{class_name}' ({class_value}): {len(feature_values_class)}")
    plt.hist(
        feature_values_class, 
        bins=20, 
        alpha=0.7, 
        color=color, 
        label=class_name, 
        density=True
    )


# Convert labels array for the new data
labels_array_new = np.array(labels2) 
# Extract the same feature index from the new data
feature_values_HC = indp_withht[:, feature_index]

# If "HC" corresponds to label == 0 in the new labels array, filter by that
feature_values_class_HC = feature_values_HC[labels_array_new == 0]

print(f"Feature values in class 'HC': {len(feature_values_class_HC)}")
plt.hist(
    feature_values_class_HC,
    bins=20,
    alpha=0.7,
    color='purple',
    label='HC',
    density=True
)

plt.xlabel('L Cereb 3 - L Cereb 4-5 FC', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=18)
plt.grid(False)
plt.show()

#%% Boxplots comparing models (Lasso, SVM, RFE, RF)

data = [rsq_las_01[:500], rsq_svm, rsq_rfe_ex, rsq_rfs_ex]
labels_box = ['Lasso', 'SVM', 'RFE', 'RF']

def significance_stars(p):
    """Convert p-value into significance stars."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

# T-tests between Lasso and others
p_values = []
for i in range(1, len(data)):
    _, p = ttest_ind(data[0], data[i], equal_var=False)
    p_values.append(p)

df = pd.DataFrame({
    'OLS Adj. R-squared': np.concatenate(data),
    'Model': np.repeat(labels_box, [len(d) for d in data])
})

sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(
    x='Model', 
    y='OLS Adj. R-squared', 
    data=df, 
    palette="Blues", 
    width=0.5,
    medianprops={"color": "black"}, 
    boxprops={"edgecolor": "black"}, 
    whiskerprops={"color": "black"}, 
    capprops={"color": "black"},
    ax=ax
)
ax.set_xlabel('Feature Extraction Model', fontsize=16)
ax.set_ylabel('OLS Adj. R-squared', fontsize=16)
ax.tick_params(labelsize=15)
sns.despine(trim=True)

y_max = df['OLS Adj. R-squared'].max()
h = 0.02
line_offsets = [1.0, 1.15, 1.3]

for i, p_val in enumerate(p_values, start=1):
    sig_label = significance_stars(p_val)
    x1, x2 = 0, i
    y = y_max * line_offsets[i-1]
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((x1+x2)*0.5, y+h, sig_label, ha='center', va='bottom', fontsize=16)

plt.tight_layout()
plt.show()

#%% Boxplots comparing clinical vs clinical+fMRI
# Example data:
# rsq_las_01_cl and rsq_las_01 are assumed to be arrays of adjusted R-squared values
from scipy import stats

data = [rsq_las_cl, adj_r2_ho_tot_01[:100]]
labels_box2 = ['Clinical', 'Clinical + fMRI']

def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'
    
t_stat, p_value = stats.ttest_ind(rsq_las_cl, adj_r2_ho_tot_01[:100], equal_var=False)
print("P-value:", p_value)
sig_label = significance_stars(p_value)

sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(
    data=data, 
    orient='v', 
    width=0.5, 
    palette="Blues", 
    medianprops={"color": "black"}, 
    boxprops={"edgecolor": "black"}, 
    whiskerprops={"color": "black"}, 
    capprops={"color": "black"}, 
    ax=ax
)
ax.set_xticklabels(labels_box2, fontsize=16)
ax.set_ylabel('OLS Adj. R-squared', fontsize=18)
ax.tick_params(labelsize=19)
sns.despine(trim=True)

y_max = max(np.max(rsq_las_cl), np.max(adj_r2_ho_tot_01[:100]))
h = 0.02
y = y_max * 1.05
x1, x2 = 0, 1
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
ax.text((x1+x2)*0.5, y+h, sig_label, ha='center', va='bottom', fontsize=16)
plt.tight_layout()
plt.show()

#%%

#%% Compare alpha values for LASSO
x_labels = ['0.001', '0.005', '0.01', '0.05', '0.07', '0.1', '0.2', '0.5', '1']

def pad_list_with_zeros(input_list, target_size=500):
    """Pads a list with zeros until it reaches the target size."""
    current_size = len(input_list)
    if current_size < target_size:
        zeros_to_add = target_size - current_size
        input_list.extend([0] * zeros_to_add)
    return input_list

# unconverged_count_0001, etc. are assumed defined globally
unconv_0001 = np.where(np.array(unconverged_count_0001) == 0)[0]
unconv_0005 = np.where(np.array(unconverged_count_0005) == 0)[0]
unconv_001 = np.where(np.array(unconverged_count_001) == 0)[0]

allalp = [rsq_las_0001, rsq_las_0005, adj_r2_ho_tot_001_, adj_r2_ho_tot_005, adj_r2_ho_tot_007, adj_r2_ho_tot_fin, adj_r2_ho_tot_02, rsq_las_05, rsq_las_1]

means = [np.mean(samples) for samples in allalp]

plt.figure(figsize=(10, 6))
plt.plot(x_labels, means, marker='o', linestyle='-', color='b', label='Mean')

for i, samples in enumerate(allalp):
    colors = ['g'] * len(samples)
    if i == 0:
        for j in range(len(samples)):
            if j in unconv_0001:
                colors[j] = 'r'
    elif i == 1:
        for m in range(len(samples)):
            if m in unconv_0005:
                colors[m] = 'r'
    elif i == 2:
        for n in range(len(samples)):
            if n in unconv_001:
                colors[n] = 'r'
    if i == 2:
        label = 'Bootstrapped Sample'
    else:
        label = ''
    plt.scatter([x_labels[i]] * len(samples), samples, color=colors, alpha=0.7, label=label)

plt.xlabel('Alpha Values for LASSO', fontsize=18)
plt.ylabel('OLS Adjusted R-squared', fontsize=18)
plt.xticks(size=16)
plt.yticks(size=16)
plt.grid(False)
plt.legend(fontsize=17)
plt.show()


#%% Plotting a connectivity matrix with given connections and coefficients
connections = [
    'Ventral-Pallidum l and Postcentral l',
    'Cerebelum-6 r and Postcentral l',
    'Dentate-Nucleus r and Red-Nucleus l',
    'Putamen l and Cerebelum-45 r',
    'Cerebelum-45 l and Subthalamic-Nucleus l'
]

coefficients = [0.31794684, -0.11179043, -0.187865, 0.21198384, 0.20613551]

region_to_index = {
    'Ventral-Pallidum l': 0,
    'Postcentral l': 1,
    'Cerebelum-6 r': 2,
    'Dentate-Nucleus r': 3,
    'Red-Nucleus l': 4,
    'Putamen l': 5,
    'Cerebelum-45 r': 6,
    'Cerebelum-45 l': 7,
    'Subthalamic-Nucleus l': 8
}

matrix_size = len(region_to_index)
matrix = np.zeros((matrix_size, matrix_size))

# Fill the matrix with given coefficients
for i, connection in enumerate(connections):
    region1, region2 = connection.split(' and ')
    index1 = region_to_index[region1]
    index2 = region_to_index[region2]
    matrix[index1, index2] = coefficients[i]
    matrix[index2, index1] = coefficients[i]

plt.figure(figsize=(10, 10))
masked_matrix = np.ma.masked_where(matrix == 0, matrix)
heatmap = plt.imshow(masked_matrix, cmap='coolwarm', interpolation='none')
cbar = plt.colorbar(heatmap)
cbar.set_label('Regression Coefficient', fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.xticks(ticks=np.arange(matrix_size), labels=region_to_index.keys(), rotation=90, fontsize=22)
plt.yticks(ticks=np.arange(matrix_size), labels=region_to_index.keys(), fontsize=22)
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.gca().set_facecolor('white')
plt.show()

#%%
    
# Redefine the variables since execution state was reset

connections = [
    'Putamen l and Cerebelum-45 r',
    'Precentral l and Supp-Motor-Area r',
    'Ventral-Pallidum l and Postcentral l',
    'Postcentral r and Subthalamic-Nucleus r',
    'Dentate-Nucleus r and Red-Nucleus l',
    'Cerebelum-3 l and Cerebelum-45 l',
    'Cerebelum-45 l and Postcentral l',
    'Cerebelum-45 l and Subthalamic-Nucleus l',
    'Thalamus l and Putamen r',
    'Substantia-Nigra-pc r and Supp-Motor-Area r',
    'Cerebelum-6 r and Postcentral l',
    'Cerebelum-6 r and Dentate-Nucleus r',
    'Substantia-Nigra-pc r and Cerebelum-3 r'
]

coefficients = [
    0.33559636, -0.12539072,  0.21316444,  0.12843726, -0.15942668,
    0.1054737 ,  0.09844194, -0.13237889, -0.08039606,  0.12314074,
    0.19065834, -0.18556471, -0.06475955
]

region_to_index = {
 'Ventral-Pallidum l': 0,
 'Substantia-Nigra-pc r': 1,
 'Thalamus l': 2,
 'Putamen r': 3,
 'Putamen l': 4,
 'Cerebelum-3 l': 5,
 'Cerebelum-3 r': 6,
 'Cerebelum-45 l': 7,
 'Cerebelum-45 r': 8,
 'Cerebelum-6 r': 9,
 'Dentate-Nucleus r': 10,
 'Postcentral l': 11,
 'Postcentral r': 12,
 'Precentral l': 13,
 'Supp-Motor-Area r': 14,
 'Subthalamic-Nucleus l': 15,
 'Subthalamic-Nucleus r': 16,
 'Red-Nucleus l': 17
}


matrix_size = len(region_to_index)
matrix = np.zeros((matrix_size, matrix_size))

# Fill the matrix with given coefficients
for i, connection in enumerate(connections):
    region1, region2 = connection.split(' and ')
    index1 = region_to_index[region1]
    index2 = region_to_index[region2]
    matrix[index1, index2] = coefficients[i]
    matrix[index2, index1] = coefficients[i]

plt.figure(figsize=(20, 10))
masked_matrix = np.ma.masked_where(matrix == 0, matrix)
heatmap = plt.imshow(masked_matrix, cmap='coolwarm', interpolation='none')
cbar = plt.colorbar(heatmap)
cbar.set_label('Regression Coefficient', fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.xticks(ticks=np.arange(matrix_size), labels=region_to_index.keys(), rotation=90, fontsize=22)
plt.yticks(ticks=np.arange(matrix_size), labels=region_to_index.keys(), fontsize=22)
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.gca().set_facecolor('white')
plt.show()
    