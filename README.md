# DBS Outcome Prediction Using Resting-State fMRI Connectivity

This repository includes anonymized data and code for doing feature extraction on resting-state fMRI ROI-ROI Pearson's correlations. The code is written in Python. The repository is part of ongoing research in the Radiology-Morrison-lab-UCSF.

## Contents

* `allrois_fc_mar.mat`:
  * This .mat file contains all subjects' ROI-based functional connectivity matrix. The ROIs are extracted using an MNI-based (Zhang et. al. 2018) atlas. Feature extraction is conducted on this matrix.
* `dataJAN.xlsx`:
   * This .xlsx file is the master deidentified spreadsheet that contains clinical data for all patients. These data include clinical covariates such as age, sex, etc. in addition to the LEDD scores. All dates (e.g., surgeries and follow-up visits) have been altered, but the time intervals between events remain consistent to ensure accurate calculations.
* `mappin3.xlsx`:
   * This .xlsx file contains the information needed to map patient clinical IDs to their CONN toolbox ID post-preprocessing.
* `DBS-Pred.py`:
   * This .py file contains the Python code used for data reorganization and analysis and for plotting the results.
* `generate_conn.m`:
   * This .m script contains the code used to generate the functional connectivity matrix file from individual subjects' ROI timeseries.
* `Ex_Subj_ROI.mat`:
   * This .mat file contains a deidentified example ROI time series file for one of the patients.

## Licenses

* Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

## Acknowledgments

This project is part of ongoing research in the Radiology-Morrison-lab-UCSF. Special thanks to all contributors and collaborators involved in this work.
