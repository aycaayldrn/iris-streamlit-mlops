FROM python:3.9-slim

#set the working directory inside the container
WORKDIR /app

#Copy requirements and install dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy the trained model and the app folder
COPY model.joblib .
COPY app/ app/

EXPOSE 8501

#Command to run
CMD ["streamlit", "run", "app/app.py"]