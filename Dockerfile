FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

RUN python manage.py collectstatic --no-input

CMD ["sh", "-c", "python manage.py migrate && gunicorn --bind 0.0.0.0:7860 config.wsgi:application"]