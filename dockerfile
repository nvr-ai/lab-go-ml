FROM alpine:3.
RUN apk add --no-cache \
    vips-dev \
    vips-tools \
    vips-dev-doc \
    vips-dev-doc-html \
    vips-dev-doc-pdf \
    vips-dev-doc-man \
    vips-dev-doc-info \
    vips-dev-doc-txt \
    vips-dev-doc-xml \