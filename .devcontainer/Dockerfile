ARG VARIANT="3.8"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# requirements
COPY src/requirements.txt /tmp/pip-tmp/requirements-src.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-src.txt \
    && rm -rf /tmp/pip-tmp

# requirements for test
COPY tests/requirements.txt /tmp/pip-tmp/requirements-tests.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-tests.txt \
    && rm -rf /tmp/pip-tmp

# requirements for visual studio code
COPY .vscode/requirements.txt /tmp/pip-tmp/requirements-vscode.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-vscode.txt \
    && rm -rf /tmp/pip-tmp
