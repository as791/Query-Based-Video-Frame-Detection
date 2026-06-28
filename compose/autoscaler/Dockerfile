FROM docker:27-cli

RUN apk add --no-cache redis

COPY autoscaler.sh /usr/local/bin/autoscaler.sh

ENTRYPOINT ["/bin/sh", "/usr/local/bin/autoscaler.sh"]
