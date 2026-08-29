# hash:sha256:56d5f619d8d37f44a6452756beef82f2e238602a975030db701086981b38ba8a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/swdb:Python3.12code-server4.133bedrock-auth0.5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN conda install -y \
        jupyterlab \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-access-api \
    hdmf-zarr \
    matplotlib \
    numpy \
    pandas \
    pynwb \
    scipy
