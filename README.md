# Heart Diseases Diagnosis Web App

The project is based on heart disease classification model. 
The data is retrieved from [Kaggle-Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

The app requests age, sex, chest pain type, serum cholestoral, maximum heart rate and number of major vessels, then returns patient's condition.


## Modeling

After EDA, Feature Engineering (One Hot Encoding and Standardiztion) and Model Selection; XGBoostClassifier is selected considering metrics. 
The detailed modeling processes are mentioned in notebook. Moreover, if you are curious about architecture, you can find the scheme diagram at '/extra/heart-disease-architecture.png'.
## Installation

You can clone the repo. Then you need to follow some steps;

You can download [Docker](https://docs.docker.com/engine/install/) according to your OS. If you don't want, I will leave the usage manual without using Docker (but Docker is highly recommended).

## Usage

### 1) without Docker (Manual)

The repo involves two .py files; model.py and predict.py. Model.py generates pickle files of xgb, ohe and standardization models. The models already exist in "models/" folder, but if you want to change dataset or to see how it works, just run model.py.
```
python model.py
```

In the predict.py, users' inputs are processed to predict heart disease risk. [Flask](https://flask.palletsprojects.com/en/2.2.x/) is used to take data and show the results. 
```
python predict.py
```
While the web app is running, you can be routed to "/welcome" and "/predict" pages. 

localhost:5000/welcome => User can enter the required field for prediction.

localhost:5000/predict => The outcome is shared.


### 2) Docker

The Docker sets up an environment including all the requirements with the right versions.

You just need to build an docker image.
```
docker build -t <name-of-image> .
```

When the built is done, docker image is ready to run.

```
docker run -it -p 5000:5000 -v $(pwd):/app 
```
Now you can connect 'localhost:5000/welcome' on your browser to test app.

