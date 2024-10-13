
## Getting Started

clone repo

```sh
git clone git@github.com/humankernel/rag
cd rag
```

setup python env

```sh
python -m venv .venv
source .venv/bin/activate.fish

pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## pip uci repo

```
# linux: ~/.config/pip/pip.conf
# windows: ~\AppData\Roaming\pip\pip.ini

[global]
timeout = 120
index = http://nexus.prod.uci.cu/repository/pypi-all/pypi
index-url = http://nexus.prod.uci.cu/repository/pypi-all/simple
[install]
trusted-host = nexus.prod.uci.cu