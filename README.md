# Audio-Based-Anomaly-Detection-for-Industrial-Machinery-End-to-End-Project-using-MLflow-DVC
An innovative machine learning model focused on real-time anomaly detection in audio recordings of industrial pumps.

## Project Overview
The essence of this project revolves around crafting and fine-tuning a deep-learning model that can discern between normal and anomalous operational sounds from industrial pumps. We utilize the specialized pump sound subset of the MIMII Dataset, which is comprised of diverse acoustic signatures representing various operational states of industrial pumps.

## Dataset Description
The pump sound dataset is a segment of the MIMII Dataset, featuring acoustic recordings that capture the operational nuances of industrial pumps. It includes a rich tapestry of soundscapes illustrating normal functionality, as well as a series of anomalies. These recordings are complemented by ambient factory noises, rendering a realistic backdrop for the model's training and evaluation.

This project is a continuation of the Jupiter notebooks I shared in Kaggle where I used supervised learning and unsupervised learning techniques respectively.
1. [Industrial Pump Anomaly Detection using DL](https://www.kaggle.com/code/jaison14/industrial-pump-anomaly-detection-using-dl)
2. [Anomaly Detection using Autoencoder](https://www.kaggle.com/code/jaison14/anomaly-detection-using-autoencoder)

## Key Accomplishments:
This project has consistently demonstrated a high F1 score, signifying its capability to balance precision and recall effectively across various testing scenarios.
* In the test phase, the model attained near-perfect accuracy, showcasing its adeptness at anomaly detection within the complex auditory domain of industrial pumps.
* Using supervised learning, achieved exceptional model performance in testing (Accuracy: 99%, Precision: 96.55%, Recall: 100%, F1 Score: 98.25%, AUC-ROC: 99.34%) — showcasing the potential for high-impact utility in predictive maintenance and efficiency identification.
* Focused on identifying unknown anomalous sounds under conditions where only normal operational sounds have been provided for training. Used encoders for unsupervised learning showing  great performance in testing((Accuracy: 96.3%, Precision: 97.8%, Recall: 96.4%, F1 Score: 97.1%).
* Prepared an MLFLow model ready to be deployed at the Azure batch endpoint for real-time batch inference.

## Screenshots from dagshub for Training of Encoder
![image](https://github.com/JAISON14/Audio-Based-Anomaly-Detection-for-Industrial-Machinery-End-to-End-Project-using-MLflow-DVC/assets/24632348/dcdcd2cd-c93c-4835-883e-9a032db25072)
![image](https://github.com/JAISON14/Audio-Based-Anomaly-Detection-for-Industrial-Machinery-End-to-End-Project-using-MLflow-DVC/assets/24632348/bbadaa01-d507-4cd0-882c-56a417598b7d)

## Conclusion
This project epitomizes the fusion of acoustic analytics and machine learning, paving the way for proactive maintenance strategies that safeguard the operational health of industrial pumps. This endeavor not only enhances reliability but also serves as a stepping stone toward the automation of industrial monitoring systems.



