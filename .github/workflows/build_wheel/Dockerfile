ARG BUILD_IMAGE
FROM ${BUILD_IMAGE} AS builder
WORKDIR /libosstorchconnector
COPY . .
ARG PYTHON_VERSION
ENV PY_VER=${PYTHON_VERSION}
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    export PATH="/root/miniconda3/bin:$PATH" && \
    conda activate py${PY_VER} && \
    echo -e "[build_ext]\nlibrary_path=oss_connector.cpython-${PY_VER}-x86_64-linux-gnu.so" > setup.cfg && \
    python3 -u setup.py bdist_wheel

FROM scratch AS release
COPY --from=builder /libosstorchconnector/dist/osstorchconnector* /
