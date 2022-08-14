 #Using the base image with latest python version
 FROM python:3.8

 COPY requirements.txt .
 
 # Install dependencies
 RUN pip install -r requirements.txt

 COPY . .

 #Starting the python application
 EXPOSE 5000

 RUN python model.py

 CMD ["python3", "predict.py"]