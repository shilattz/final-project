# Final Project: Faulty Sensor Detection in Aircraft Monitoring

This repository contains our final-year Software Engineering project focused on detecting faulty sensors in aircraft monitoring systems using **Data Science** and **Machine Learning** methods.

---

## Project Goal

The goal of this project is to automatically detect which sensor in a multi-sensor aircraft monitoring system is **faulty**, based on inconsistent or noisy readings.  
Since real sensor observations data is unavailable due to confidentiality, we developed a **simulation-based dataset** that generates realistic flight trajectories and sensor observations, including **controlled noise and injected faults**.

---

## Project Overview

The project initially began with **one real flight path** collected from the [OpenSky Network](https://opensky-network.org/) API.  
We used this path as a basis for developing a **synthetic flight simulator**, which allowed us to expand the dataset and train models on a much wider variety of trajectories.

- **400 synthetic flight paths** generated with realistic altitude, velocity, and angular dynamics  
- **5 virtual sensors** (GPS and Angular) surrounding each trajectory  
- Controlled **fault injection** where one sensor per example becomes faulty  
- Rich **feature engineering** (deviation, standard deviation, slope, median difference)  
- Training and evaluation of models to classify the faulty sensor  
- Full **experiment tracking and visualization** using ClearML

---

## Technologies Used

- **Python 3**
- **PyTorch** – Transformer-based deep learning model  
- **XGBoost** – Baseline statistical model  
- **Scikit-learn**, **Pandas**, **NumPy**  
- **ClearML** – Experiment management and results tracking  
- **Matplotlib**, **Seaborn** – Visualization of sensor patterns and metrics  

---

## Results

- **XGBoost** reached ~83% route-level accuracy  
- **Transformer** achieved ~75% after retraining on the expanded dataset  
- The results highlight that both approaches can detect faulty sensors reliably, and that the Transformer shows potential for scaling to larger real-world datasets.

---

## Contributors

- **Ayelet Cohen** 
- **Shilat Tzur**   

---

> _Developed as part of the Software Engineering Final Project at Azrieli College of Engineering (2025)._
