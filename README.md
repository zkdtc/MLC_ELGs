# Machine Learning Classifiers for intermediate redshift Emission Line Galaxies

## Introduction
Classification of intermediate redshift (z = 0.3–0.8) emission line galaxies as star-forming galaxies, composite galaxies, active galactic nuclei (AGN), or low-ionization nuclear emission regions (LINERs) using optical spectra alone was impossible because the lines used for standard optical diagnostic diagrams: [NII], Hα, and [SII] are redshifted out of the observed wavelength range. In this work, we address this problem using four supervised machine learning classification algorithms: k-nearest neighbors (KNN), support vector classifier (SVC), random forest (RF), and a multi-layer perceptron (MLP) neural network. For input features, we use properties that can be measured from optical galaxy spectra out to z < 0.8—[O III]/Hβ, [O II]/Hβ, [O III] line width, and stellar velocity dispersion—and four colors (u−g, g−r, r−i, and i−z) corrected to z = 0.1. The labels for the low redshift emission line galaxy training set are determined using standard optical diagnostic diagrams. RF has the best area under curve (AUC) score for classifying all four galaxy types, meaning highest distinguishing power. Both the AUC scores and accuracies of the other algorithms are ordered as MLP>SVC>KNN. The classification accuracies with all eight features (and the four spectroscopically- determined features only) are 93.4% (92.3%) for star-forming galaxies, 69.4% (63.7%) for composite galaxies, 71.8% (67.3%) for AGNs, and 65.7% (60.8%) for LINERs. The stacked spectrum of galaxies of the same type as determined by optical diagnostic diagrams at low redshift and RF at intermediate redshift are broadly consistent. Our publicly available codea and trained models will be instrumental for classifying emission line galaxies in upcoming wide-field spectroscopic surveys.

## Files
### data_matched_step2_newz_sm.csv 
File containing the input parameters for training. 
### data_elg.csv 
File containing the input parameters for test sample used by scikit_kfold_classifier.py. 
### eboss-elg-classification.fits 
File containing 0.32<z<0.8 emission line galaxies MJD, Plate, FIBERID, z, [O III]/Hβ, [O II]/Hβ, [O III] line width, and stellar velocity dispersion—and four colors (u−g, g−r, r−i, and i−z) corrected to z = 0.1 and classification result. 1=SFG, 2=Composite, 3=AGN, 4=LINER. Value added catalog for SDSS-IV DR16 [A et al. 2019](https://github.com/zkdtc/MLC_ELGs).
### eboss-elg-classification.html
Data Model for eboss-elg-classification.fits

## Codes
### scikit_kfold_training.py
The code for training the model. You you select the model to use from: knn, svc, mlp, rf, rf4, rf2. knn=k-nearest neighbors, svc=suport vector classifier, mlp=multi-layer perceptron neural net work, rf=random forest, rf4=random forest using 4 spectroscopic features: [O III]/Hβ, [O II]/Hβ, [O III] line width, and stellar velocity dispersion, rf2=random forest using only [O III]/Hβ and [O III] line width. 

>Run the code as: python3 scikit_kfold_training.py 'rf'. 

The models trained will be output to the directory. 

### scikit_kfold_classifier.py
Use a trained model to make classifications to 0.32<z<0.8 emission line galaxies.

## Citation
If you use the data or code from this project, please cite [Zhang et al. 2019](https://github.com/zkdtc/MLC_ELGs).
