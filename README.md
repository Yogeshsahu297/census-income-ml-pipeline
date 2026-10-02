# Census Income Prediction – Scalable ML Pipeline in Production

A complete ML pipeline that trains a classifier on the US Census "Adult"
dataset, serves it through a **FastAPI** REST API, tests it, and ships it with
**GitHub Actions** continuous integration and **Render** continuous deployment.

| Item | Value |
|------|-------|
| Repository platform | **GitHub** |
| Repository URL | `https://github.com/Yogeshsahu297/census-income-ml-pipeline` |
| Live API URL | `https://census-income-ml-pipeline.onrender.com` |
| Python version | 3.12.3 (`.python-version`, used by CI and Render) |

> If you use Azure DevOps instead of GitHub, see [Azure DevOps variant](#azure-devops-variant).

## Project structure

```
.
├── .github/workflows/ci.yml   # CI (flake8 + pytest) and CD (Render deploy hook)
├── data/census.csv            # Census dataset
├── model/                     # model.pkl, encoder.pkl, lb.pkl  (committed on purpose)
├── screenshots/               # YOU add: continuous_integration.png, example.png, ...
├── starter/
│   ├── config.py              # paths, feature lists, random seed
│   ├── train_model.py         # training script (also writes slice_output.txt)
│   └── ml/
│       ├── data.py            # process_data (one-hot encoding + label binarizer)
│       └── model.py           # train_model, inference, metrics, save/load, slice metrics
├── tests/                     # 16 tests: ML units, data, training script, API
├── main.py                    # FastAPI app (GET / and POST /predict)
├── live_post.py               # POST request to the deployed API
├── slice_output.txt           # metrics for every slice of every categorical feature
├── model_card.md              # completed model card
├── render.yaml                # optional Render blueprint
├── requirements.txt           # pinned dependencies
├── sanitycheck.py             # Udacity helper (unchanged)
├── .flake8, pytest.ini, setup.py, .python-version, .gitignore, .gitattributes
└── README.md
```

## Setup (Windows / PowerShell)

Install Python **3.12** (https://www.python.org/downloads/) and Git, then:

```powershell
cd path\to\project
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

> Keep the pinned versions. The committed pickles were created with
> scikit-learn 1.8.0 and must be loaded with the same version.
> If you develop on a different Python version, change it in `.python-version`
> (CI and Render read it) and in `render.yaml`.

## Run

```powershell
# 1. Train (creates model/*.pkl and slice_output.txt; ~10 s)
python starter/train_model.py

# 2. Lint + tests (what CI runs)
flake8 .
pytest -v

# 3. Rubric helper – when prompted, type:  tests/test_api.py
python sanitycheck.py

# 4. Start the API locally
uvicorn main:app --reload
#    http://127.0.0.1:8000/       -> greeting
#    http://127.0.0.1:8000/docs   -> Swagger UI with the request example
```

Example request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"age\":45,\"workclass\":\"Private\",\"fnlgt\":160000,\"education\":\"Masters\",\"education-num\":14,\"marital-status\":\"Married-civ-spouse\",\"occupation\":\"Exec-managerial\",\"relationship\":\"Husband\",\"race\":\"White\",\"sex\":\"Male\",\"capital-gain\":15024,\"capital-loss\":0,\"hours-per-week\":50,\"native-country\":\"United-States\"}"
# {"prediction":">50K"}
```

## Model

* `RandomForestClassifier`, 80/20 stratified train/test split (seed 42).
* Test metrics: precision 0.7919, recall 0.5995, F1 0.6824. See `model_card.md`.
* `python starter/train_model.py` runs `compute_slice_metrics` for every
  categorical feature and writes `slice_output.txt`.
* **Model artifacts are required for submission.** `.gitignore` ignores `*.pkl`
  but contains explicit exceptions for `model/model.pkl`, `model/encoder.pkl`
  and `model/lb.pkl`. Verify with `git ls-files model` – all three must be listed.
  If not: `git add -f model/model.pkl model/encoder.pkl model/lb.pkl`.
  The API loads them directly, so inference works without retraining.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | `{"greeting": "Welcome to the Census Income Prediction API!"}` |
| POST | `/predict` | Body: one census record (Pydantic model `CensusRecord`, example included) → `{"prediction": "<=50K"}` or `{"prediction": ">50K"}` |

Hyphenated feature names (`education-num`, `marital-status`, ...) are Pydantic
aliases, so the JSON body uses the same names as the dataset.

## Tests

`pytest` runs 16 tests (≥ 6 required): unit tests for training/inference/metrics/
save-load/slices/data processing/training script, and API tests: a GET test
(status code **and** body), one POST test per possible prediction (`>50K` and
`<=50K`), a 422 validation test and a test that every field has an example.
Coverage is ~99 % (`pytest --cov=main --cov=starter --cov-report=term-missing`).

## CI/CD (GitHub Actions + Render)

`.github/workflows/ci.yml` runs on every push to `main`/`master`:

1. **test job** – Python from `.python-version`, `pip install -r requirements.txt`,
   `flake8 .`, `pytest`.
2. **deploy job** – `needs: test`, only for pushes to `main`/`master`: POSTs to the
   Render deploy hook stored in the GitHub secret `RENDER_DEPLOY_HOOK_URL`.
   No credentials are in the repository.

---

## What YOU must do manually (needs your accounts)

### A. Create the GitHub repo and push

```powershell
git init -b main
git add -A
git status            # confirm model/*.pkl and data/census.csv are listed
git commit -m "Initial commit: census income ML pipeline"
git remote add origin https://github.com/Yogeshsahu297/census-income-ml-pipeline.git
git push -u origin main
```
Use a public repo (or one the reviewer can access). Put its URL in the table at
the top of this README. Authenticate with the browser/Git Credential Manager –
never put a token in a file or in the remote URL.

### B. Protect `main` (rubric: deploy from the protected branch)
GitHub → Settings → Branches → *Add branch protection rule* for `main`
→ tick **Require status checks to pass before merging** and select
`Continuous integration (flake8 + pytest)`.

### C. Create the Render web service
1. Render → New → **Web Service** → connect your GitHub repo, branch `main`.
2. Runtime **Python 3**; Build command `pip install -r requirements.txt`;
   Start command `uvicorn main:app --host 0.0.0.0 --port $PORT`.
   Add env var `PYTHON_VERSION` = `3.12.3`.
3. Settings → **Auto-Deploy: Off** (deploys are triggered by the CI job only after
   tests pass; this guarantees "deploy only after pytest and flake8 pass").
   *(Alternative: leave Auto-Deploy on with “After CI Checks Pass”, and
   delete the `deploy` job from the workflow.)*
4. Settings → **Deploy Hook** → copy the URL.
5. GitHub repo → Settings → Secrets and variables → Actions → **New repository
   secret**: name `RENDER_DEPLOY_HOOK_URL`, value = the hook URL.
6. Push any commit to `main`; both CI jobs should turn green and Render deploys.

### D. Take the screenshots (save into `screenshots/`, exact names)

| File | What it must show |
|------|-------------------|
| `continuous_integration.png` | GitHub → Actions → the green run showing the `test` job (flake8 + pytest) passing |
| `example.png` | `http://127.0.0.1:8000/docs` with `POST /predict` expanded, showing the example request body |
| `continuous_deployment.png` | The green `deploy` job in Actions (and/or Render showing the deploy succeeded / auto-deploy setting). **Hide the hook URL** |
| `live_get.png` | Browser at `https://census-income-ml-pipeline.onrender.com/` showing the greeting JSON (URL bar visible) |
| `live_post.png` | Terminal after `python live_post.py https://census-income-ml-pipeline.onrender.com` showing the status code and prediction |

Free Render services sleep when idle; the first request can take ~1 minute.
Then `git add screenshots && git commit -m "Add screenshots" && git push`.
(Screenshots are *not* ignored by `.gitignore` in this project.)

### E. Final checks before submitting
```powershell
flake8 .            # no output = pass
pytest              # 16 passed
git ls-files model  # model.pkl, encoder.pkl, lb.pkl (+ .gitkeep)
git log -p | Select-String -Pattern "hooks|token|secret" # nothing sensitive
```
Submit the GitHub repo (and keep the URL in this README), or zip the folder
(include `.git`, `model/`, `screenshots/`).

## Azure DevOps variant

The rubric also allows Azure Pipelines. To use it, create a private Azure Repos
repo, push this project to it, and add an `azure-pipelines.yml` with a CI stage
(`UsePythonVersion@0` 3.12, `pip install -r requirements.txt`, `flake8 .`,
`pytest`) and a deploy stage with `dependsOn: CI` and
`condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))`
that calls the Render deploy hook stored as a **secret pipeline variable**.
Record the Azure Repos URL here, take `continuous_integration.png` from the
pipeline run, and submit a ZIP of a **fresh clone** (`git clone <url> fresh`)
that includes `.git` – check `.git/config` contains no token first.
The GitHub workflow in this repo is ignored by Azure.

## Optional extras (“make it stand out”)
Coverage is measured with `pytest-cov` (~99 %). Possible additions: Codecov
upload, API-key authentication, an HTML front end.

## License
See `LICENSE.txt`.
