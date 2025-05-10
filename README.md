# nlp-data-assessment

## Prerequisites

- Docker - If on MacOS Install Docker 4.37.2 or latest from https://docs.docker.com/desktop/release-notes/#4372 to avoid having Docker startup issues
- Python 3.9 
- Git - Install from https://git-scm.com/downloads

## Setup Instructions

### Project Structure
<img width="760" alt="image" src="https://github.com/user-attachments/assets/9fd12a40-c333-4d40-8011-3945d3d9115b" />


> **Note:** The LSTM and CLSTM model are commented out in the model_training.py file for the Docker build as these models generally require significant computational resources for faster training. I trained them on Google Colab for faster training with GPU acceleration. You can uncomment these functions `clstm_model` and `lstm_model` in `model_training.py` file if you have adequate computational resources. Link to the Google colab file: https://colab.research.google.com/drive/18U4zCa04L9s8w2GJqoSUBykQDqwwdNW3#scrollTo=OQknX1uGyrAx&uniqifier=1

## Run Using Docker...

### Run Using Docker (Recommended)

#### Clone the repository:
   - ```git clone https://github.com/MNCEDISIMNCWABE/nlp-data-assessment.git```
   - ```cd nlp-data-assessment```
  
#### Data Setup
- Place `patient_behavior_data.csv` file in the project root directory after cloning the repo
- Expected CSV columns: patient_id,name,surname,gender,medication,dose,bmi,weight,height,systolic,diastolic,concentration,impulsivity,distractibility,hyperactivity,sleep,mood,appetite,doctor_notes

#### Build the Docker image:
```docker build -t nlp-app .```

#### Run the Docker Container
```docker run nlp-app```


