FROM busybox:1.33.1

# Images generated from this Dockerfile just exist to manage docs site content
# without needing a separate way of storing and versioning it. They copy the
# site content into /volumes/content and assume some other container is set up
# to mount that volume and serve the content.
COPY /public/ /content/
ENTRYPOINT ["sh", "-c", "cp -rf /content/* /volumes/content/"]
