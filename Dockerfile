FROM mambaorg/micromamba:1.5.8

# Instalar en el entorno "base": es el que activa el entrypoint de la imagen
# (la variable que respeta micromamba es ENV_NAME, no CONDA_ENV)
RUN micromamba install -y -n base -c conda-forge \
    python=3.10 \
    fenics-dolfinx \
    gmsh \
    python-gmsh \
    numpy \
    scipy \
    matplotlib \
    petsc4py \
    mpi4py \
    pytest \
    pytest-cov \
    && micromamba clean -a -y

WORKDIR /workspace/psyop
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/psyop

# Activar el entorno base durante los RUN de build
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-deps -e .

CMD ["psyop", "--test"]
