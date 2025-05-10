# nlp-data-assessment

## Prerequisites

- Docker - If on MacOS Install Docker 4.37.2 or latest from https://docs.docker.com/desktop/release-notes/#4372 to avoid having Docker startup issues
- Python 3.9 
- Git - Install from https://git-scm.com/downloads

## Setup Instructions

### Project Structure
.
├── Dockerfile                  
├── README.md                  
├── requirements.txt           
├── model_training.py  
├──model_training.ipynb
└── patient_behavior_data.csv   # data (excluded from repo, add to project root directory after cloning)

### Run Using Docker (Recommended)

1. **Clone the repository**:
   - ```git clone https://github.com/MNCEDISIMNCWABE/nlp-data-assessment.git```
   - ```cd nlp-data-assessment```
  
2. ## Data Setup
- Place `patient_behavior_data.csv` file in the project root directory after cloning the repo
- Expected CSV columns: patient_id,name,surname,gender,medication,dose,bmi,weight,height,systolic,diastolic,concentration,impulsivity,distractibility,hyperactivity,sleep,mood,appetite,doctor_notes

3. **Build the Docker image**:
```docker build -t nlp-app .```

2. **Run the Docker Container**:
```docker run nlp-app```


