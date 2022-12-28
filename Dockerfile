FROM ubuntu:focal-20221130
RUN apt update
RUN apt install git -y
RUN git clone https://github.com/MLMonkATGY/segment_ocr_payslip.git
WORKDIR /segment_ocr_payslip
RUN apt install python-is-python3 -y
RUN apt install python3-pip -y
RUN pip install -r requirements.txt
RUN DEBIAN_FRONTEND=noninteractive apt install tesseract-ocr -y
RUN apt install ffmpeg libsm6 libxext6  -y
CMD ["python", "src/app/app.py"]
