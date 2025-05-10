# nlp-data-assessment

## Prerequisites

- Docker - If on MacOS Install Docker 4.37.2 or latest from https://docs.docker.com/desktop/release-notes/#4372 to avoid having Docker startup issues
- Python 3.9 
- Git - Install from https://git-scm.com/downloads

## Setup Instructions

### Project Structure
<img width="760" alt="image" src="https://github.com/user-attachments/assets/9fd12a40-c333-4d40-8011-3945d3d9115b" />

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


