FROM python:3.10
WORKDIR /microservice-breastcancer-prediction
COPY requirements.txt /microservice-breastcancer-prediction
EXPOSE 5000
RUN pip install virtualenv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir -r ./requirements.txt
COPY . /microservice-breastcancer-prediction
CMD ["python", "app.py"]