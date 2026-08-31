# hash:sha256:0da50398a0e7ff8e78522ab77816fe209e0027d0442343f553867599c18f7b9f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/swdb:Python3.12code-server4.133bedrock-auth0.5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN conda install -y \
        jupyterlab==4.6.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.10.0 \
    dynamax[notebooks]==1.0.2 \
    hdmf-zarr==0.13.0 \
    jax==0.11.1 \
    matplotlib==3.11.1 \
    numpy==2.5.2 \
    pandas==3.0.5 \
    pynwb==4.1.0 \
    scikit-learn==1.9.0 \
    scipy==1.18.1 \
    seaborn==0.13.2

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
