
Final Project: 
Faulty Sensor Detection in Aircraft Monitoring

This repository contains the final project focusing on detecting faulty sensors in aircraft trajectories using machine learning.

Project Goal
The goal of this project is to detect which sensor in a network of aircraft monitoring sensors is faulty, 
even when all sensors are reporting noisy data. We use simulated data and inject faults into one sensor per example, 
training models to identify the faulty one based on inconsistencies in readings.

Technologies Used

- Python 3
- PyTorch
- XGBoost
- Scikit-learn
- ClearML
- Pandas, NumPy
- Matplotlib, Seaborn


Highlights

- Realistic aircraft trajectory simulation
- Sensor placement and observation simulation
- Fault injection (with stronger noise in one sensor)
- Feature engineering: difference from median, standard deviation, etc.
- Classification models to predict which sensor is faulty
- Transformer-based and XGBoost models evaluated

Contributors

- Ayelet Cohen  
- Shilat Tzur

Future Work

- Improve fault injection realism  



