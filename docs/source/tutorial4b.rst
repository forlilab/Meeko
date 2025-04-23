.. _tutorial4b:

=================================================
Retrospective Docking Analysis and Model Building
=================================================

Introduction
============

Using the outcomes from the preivous tutorial, this tutorial is intended to show the basic procedure of ligand reconstruction, computing of assessment metrics (ROC-AUC, EF) in a retrospective docking analysis, and the building of regression model involving interaction vectors from AutoDock-GPU. 

The main purpose of this tutorial is to show how to build the necessary interface between packages in retrospective docking analysis and model building via a practical example. But please note that the outcomes and conclusions should not be judged professionally. In practice, thorough profiling of the dataset and feature space, careful selection of the regression model and fine-tuning the coditions, are all necessary steps to achieve the optimal model. 

Additional Dependencies
=======================

Single-Pose Molecule Reconstruction
===================================


Basic Metrics (ROC-AUC, EF) based on Single Variable
====================================================


Vectorization of Interactions, XGBoost Modeling and SHAP explanation
=====================================================================