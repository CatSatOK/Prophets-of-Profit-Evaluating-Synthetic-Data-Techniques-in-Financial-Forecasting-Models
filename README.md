# Prophets-of-Profit-Evaluating-Synthetic-Data-Techniques-in-Financial-Forecasting-Models
An comparative investigation into WGAN-GP, CTGAN, TimeGAN and DoppelGANger usage for generating synthetic time series finance data for use in forecasting model
To recreate the experiment please run the Jupyter Notebook in order (s1-s7).

s1_Data prep.ipynb to create the dataset
s2_LSTM baseline model.ipynb to create the baseline prediction model
s3_WGAN-GP.ipynb to create a synthetic dataset using WGAN-GP 
s4_CTGAN.ipynb to create a synthetic dataset using CTGAN and 
s5_TimeGAN.ipynb to create a synthetic dataset using TimeGAN 
s6_DopGAN.ipynb to create a synthetic dataset using DoppelGANger 
s7_Model_comparision to train and test the LSTM models trained on the synthetic and combination datasets & create the PCA and t-SNE plots
indicator_generator.py is functions used for data preprocessing in s1_Data prep.ipynb
