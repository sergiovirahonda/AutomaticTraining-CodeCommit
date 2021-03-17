# If you're training on a CPU based machine, use the following image:
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-0
# If you're training on a GPU enabled machine, use the following image:
#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-3
WORKDIR /root

RUN pip install pandas numpy google-cloud-storage scikit-learn opencv-python

RUN apt-get update; apt-get install git -y; apt-get install -y libgl1-mesa-dev

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone https://github.com/sergiovirahonda/AutomaticTraining-Dataset.git
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone https://github.com/sergiovirahonda/AutomaticTraining-CodeCommit.git 

RUN mv /root/AutomaticTraining-CodeCommit/model_assembly.py /root
RUN mv /root/AutomaticTraining-CodeCommit/task.py /root
RUN mv /root/AutomaticTraining-CodeCommit/data_utils.py /root
RUN mv /root/AutomaticTraining-CodeCommit/email_notifications.py /root

ENTRYPOINT ["python","task.py"]
