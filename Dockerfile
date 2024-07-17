FROM python:3.8-slim

WORKDIR /app/

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9091

ENV NAME=Assignment

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]