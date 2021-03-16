# If you're training on a CPU based machine, use the following image:
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-0
# If you're training on a GPU enabled machine, use the following image:
#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-3
WORKDIR /root

RUN pip install pandas numpy google-cloud-storage scikit-learn opencv-python

RUN apt-get update; apt-get install git -y; apt-get install -y libgl1-mesa-dev
RUN git clone https://github.com/sergiovirahonda/AutomaticTraining-Dataset.git

# If you're going to pull the code from GitHub repo
# RUN git clone https://github.com/sergiovirahonda/AutomaticTraining-BaseCode.git

# RUN mv /root/AutomaticTraining-BaseCode/model_assembly.py /root
# RUN mv /root/AutomaticTraining-BaseCode/task.py /root
# RUN mv /root/AutomaticTraining-BaseCode/data_utils.py /root
# RUN mv /root/AutomaticTraining-BaseCode/email_notifications.py /root

# If not
COPY model_assembly.py ./model_assembly.py
COPY data_utils.py ./data_utils.py
COPY task.py ./task.py
COPY email_notifications.py ./email_notifications.py
COPY AutomaticTrainingCICD-68f56bfa992e.json ./AutomaticTrainingCICD-68f56bfa992e.json

RUN export GOOGLE_APPLICATION_CREDENTIALS="/root/AutomaticTrainingCICD-68f56bfa992e.json"

ENTRYPOINT ["python","task.py"]
