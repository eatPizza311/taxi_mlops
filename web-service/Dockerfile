FROM python:3.9.16-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

# install dependencies to the system (No need virtual env)
RUN pipenv install --system --deploy

EXPOSE 4444

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:4444", "predict:app" ]

COPY ["predict.py", "lin_reg.bin", "./"]
