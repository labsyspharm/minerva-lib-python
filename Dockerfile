# Docker container to build manylinux wheels for minerva-lib-python
#
# Example:
# docker build . -t minerva-wheel
# docker run -it -v /minerva-lib-python/dist:/wheels minerva-wheel
#

FROM quay.io/pypa/manylinux2014_x86_64

WORKDIR /opt

COPY build_wheels.sh /opt

ENTRYPOINT /opt/build_wheels.sh
