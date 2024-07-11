# pull python base image
FROM python:3.11

# specify working directory
#WORKDIR 
# copy application files
#ADD . .
# WORK DIR
WORKDIR /patent_project
#ADD requirements.txt .
ADD /*.py .
ADD /*.txt .
ADD /*.yaml .
ADD /pages/*.* ./pages/
#ADD /model/* ./model/
ADD /tempDir/*.* ./tempDir/
ADD /pdfpatentchroma_db ./pdfpatentchroma_db
ADD /data/*.* ./data/

# update pip
#RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

#RUN rm *.whl

# copy application files
#ADD /heart_model/* ./app/

# expose port for application
EXPOSE 8501

# start fastapi application
ENTRYPOINT ["streamlit", "run", "welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]
