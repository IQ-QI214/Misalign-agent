FROM cua-lite/lite.osworld:latest
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libnss3-tools libnotify-bin && rm -rf /var/lib/apt/lists/*
COPY mkcert /usr/local/bin/mkcert
USER root
