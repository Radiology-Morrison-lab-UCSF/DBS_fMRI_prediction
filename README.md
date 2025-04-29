# DBS Outcome Prediction Using Resting-State fMRI Connectivity

This repository includes anonymized data and code for doing feature extraction on resting-state fMRI ROI-ROI Pearson's correlations. The code is written in Python. The repository is part of ongoing research in the Radiology-Morrison-lab-UCSF.

## Contents

* `Ex_Subj_ROI.mat`:
   * This .mat file contains an example ROI time series file for one of the patients.
 * `generate_conn.m`:
   * This .m script contains the code used to generate the functional connectivity matrix file from individual subjects' ROI timeseries.
* `allrois_fc_mar.mat`:
  * This .mat file contains all subjects' ROI-based functional connectivity matrix. The ROIs are extracted using an MNI-based (Zhang et. al. 2018) atlas. Feature extraction is conducted on this matrix.
* `data.xlsx`:
   * This .xlsx file is the master de-identified spreadsheet that contains clinical data for all patients. These data include clinical covariates such as age, sex, etc. in addition to the LEDD scores.
* `DBS-Pred.py`:
   * This .py file contains the Python code used for data reorganization and analysis and for plotting the results.
* `preop_fMRI_PD.mat`:
   * This .mat file has the CONN toolbox preprocessing template we used for the fMRI data.

## Data De-Identification

* All patients IDs are anonymous 
* All dates (e.g., surgeries and follow-up visits) in the master spreadsheet have been altered, but the time intervals between events remain consistent to ensure accurate calculations.
* The example .ROI file is de-identified.

## Licenses

* Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

## Acknowledgments

This project is part of ongoing research in the Radiology-Morrison-lab-UCSF. Special thanks to all contributors and collaborators involved in this work.
