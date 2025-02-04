# Automated-Essay-Scorer
A fine tuned deberta-v3-large model for the downstream task of scoring essays

# How to use
Make sure you have latest version of python installed and ready to use.

- Clone the repo

```git clone https://github.com/vdhkcheems/Automated-Essay-Scorer.git```

- go into the cloned dirctory and setup a python virtual environment

```python -m venv .venv```

- put it as source

```source .venv/bin/activate```

- install requirements

```pip install -r requirements.txt```

- run the app

```uvicorn app:app --reload```


