FROM mambaorg/micromamba:1.5.8

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_ENV=psyop

COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt

RUN micromamba create -y -n ${CONDA_ENV} -c conda-forge \
    python=3.10 \
    fenics \
    dolfinx \
    gmsh \
    numpy \
    scipy \
    matplotlib \
    petsc4py \
    mpi4py \
    && micromamba clean -a -y

WORKDIR /workspace/psyop
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/psyop

CMD ["python", "main.py", "--test"]
