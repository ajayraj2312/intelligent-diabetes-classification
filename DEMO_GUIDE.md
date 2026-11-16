# Professor Demo Guide

**Project:** Intelligent Diabetes Classification Using Archerfish Optimization and Machine Learning
**Format:** Local Flask application, browser demo, approximately 7 minutes.

---

## 1. Before you start (5-minute checklist)

Open a terminal in the project folder and run:

```bash
source .venv/bin/activate
python app.py
```

You should see:

```text
  GlycoSense — Intelligent Diabetes Classification
  Dashboard: http://127.0.0.1:5055
```

Open **http://127.0.0.1:5055** in your browser. Keep the terminal visible in the background in case you need to restart.

Check quickly:

- [ ] Dashboard loads with **5 records**, 3 diabetic and 2 non-diabetic.
- [ ] Model accuracy card shows **74.7%**.
- [ ] AFO card shows **8 → 4 features**.
- [ ] Sidebar shows **Random Forest** and **AFO engine active**.
- [ ] The AFO page charts load (convergence and selection frequency).
- [ ] The performance table loads all four classifiers.
- [ ] `screenshots/` is open in a folder as a backup.

If the model files are missing, run `python train_models.py` (about 20 seconds). If the server is already running from earlier, just refresh the page. If the port is busy, use `PORT=5056 python app.py`.

Do **not** demonstrate from `file://` or open the HTML templates directly — they need Flask.

---

## 2. The 7-minute script

### 0:00–1:00 — Dashboard (`/`)

Say:

> This is the clinical overview. Every assessment is stored in SQLite. The dashboard shows total patients, the diabetic/non-diabetic split, and the accuracy of the selected model on the held-out test set.
>
> The pipeline card summarises the core idea: eight clinical inputs, four features selected by Archerfish Optimization, four classifiers compared, with Random Forest selected.

Point at:

- **5 assessments** — real records created through the form.
- **74.7% model accuracy** — hold-out result, not a hard-coded number.
- **Prediction distribution** doughnut — 3 diabetic / 2 non-diabetic.
- **Classifier accuracy comparison** — all four models, conventional vs AFO.
- **AFO feature subset** — shows selected (blue) and excluded (struck-through) attributes.
- **Recent predictions** — click the arrow on any row to open its saved result.

### 1:00–2:30 — New assessment (`/predict`)

Click **New assessment**.

Say:

> The form takes the eight PIMA clinical parameters. Each field has a unit, a plausible range and a hint. Validation runs in the browser and again on the server, so the model never receives missing or impossible values.

Enter exactly:

| Field | Value |
|---|---:|
| Patient reference | `DEMO-006` |
| Pregnancies | `6` |
| Age | `50` |
| Plasma Glucose | `148` |
| Diastolic Blood Pressure | `72` |
| Triceps Skin Thickness | `35` |
| Serum Insulin | `0` |
| Body Mass Index | `33.6` |
| Diabetes Pedigree Function | `0.627` |

Say:

> I left insulin as zero, which is the dataset's missing-value marker. The app treats it as missing and imputes the training median rather than feeding a false zero to the model.

Click **Run classification**.

### 2:30–3:30 — Result (`/result/...`)

Expected output for this exact sample:

- **Diabetic**
- **57.5% diabetes probability**
- **Moderate risk** (35–60% band)
- **Random Forest**
- **4 of 8 features**
- Insulin flagged as **Imputed**
- Only Pregnancies, Glucose, BMI and DPF marked **Model input**

Say:

> This is the actual probability returned by the saved Random Forest pipeline. The gauge maps probability into low, moderate and high bands. The page also shows which AFO-selected features the model used and which original inputs were not used. The recommendation and disclaimer are always displayed because this is a screening prototype, not a diagnosis.

Optional second example — **non-diabetic, 1.2%**:

| Field | Value |
|---|---:|
| Patient reference | `DEMO-007` |
| Pregnancies | `1` |
| Age | `31` |
| Plasma Glucose | `85` |
| Blood Pressure | `66` |
| Skin Thickness | `29` |
| Insulin | `0` |
| BMI | `26.6` |
| Diabetes Pedigree Function | `0.351` |

### 3:30–5:00 — AFO report (`/afo`)

Click **AFO optimizer**.

Say:

> Archerfish Optimization is the novel part of the project. Eight features are encoded as a continuous position for each archer. A school of 22 archers takes refraction-corrected shots toward the best subset for 12 iterations. A competition term makes some archers follow a peer, and the jet dispersion cools over time to shift from exploration to exploitation. Only improving positions are retained.

Point at:

- **4/8 selected features, 50% reduction.**
- Selected list: Pregnancies, Glucose, BMI, Diabetes Pedigree Function.
- Excluded list: Blood Pressure, Skin Thickness, Insulin, Age.
- **Selection frequency:** Glucose and BMI 100%, DPF 90.9%, Pregnancies 77.3%.
- **Convergence chart:** best fitness falls from the conventional reference (0.2748) to 0.2614; the school mean falls from 0.34 to 0.26.
- **Before vs after table:** accuracy, Δ, F1 and AUC for every classifier.
- **Fitness formula** shown on the page.

Say:

> The objective is a weighted combination of cross-validated F1, cross-validated AUC and a parsimony penalty. The exact equations were not in the report, so I implemented and documented a transparent Archerfish-inspired wrapper rather than relabelling a random selector. The method, parameters and assumptions are visible on this page and in the README.
>
> The selected Random Forest matches the conventional eight-feature accuracy — 74.68% — with half the inputs, while F1 rises from 59.79% to 60.61% and AUC from 0.8128 to 0.8170.

Do not claim AFO increased accuracy; it did not on this split. Say "matched accuracy with half the features and improved F1/AUC."

### 5:00–6:00 — Model performance (`/performance`)

Click **Model performance**.

Say:

> Every metric here is computed from the untouched 154-record test split. The top row is the selected Random Forest: 74.68% accuracy, 66.67% precision, 55.56% recall, 60.61% F1 and 0.817 AUC.

Point at:

- Full comparison table: **8 configurations**, conventional vs AFO.
- Radar chart: five metrics for the four AFO models.
- ROC curves with AUC values.
- Confusion matrix: **TN 85, FP 15, FN 24, TP 30**; sensitivity 55.56%, specificity 85%.
- Impact cards: accuracy +0.0 pp, F1 +0.8 pp, AUC +0.0042, CV F1 +1.9 pp, 50% fewer features.

Say:

> I show the unfavourable rows too — AFO-SVM and AFO-KNN are slightly worse. With only 154 test records, one or two percentage points are within sampling noise. The contribution is dimensionality reduction with equal accuracy and slightly better F1/AUC for the selected model.

### 6:00–6:30 — Prediction history (`/history`)

Click **Prediction history**.

Say:

> Every prediction is persisted with its clinical inputs, probability, model and timestamp. The table supports search by patient reference or record ID, plus outcome, model and date filters, with pagination.

Type `DEMO` in the search box, choose **Diabetic**, click **Apply**. Show the filtered rows and open one record. Clear the filters afterwards.

### 6:30–7:00 — System information (`/about`)

Click **System information**.

Say:

> This page documents the problem statement, existing system limitations, proposed system, objectives, six-stage architecture, the AFO method, the four classifiers, technologies, dataset and evaluation protocol. The six architecture stages match the project report.

Finish with:

> The system is deliberately scoped as an educational screening prototype. It is not clinically validated. Future work would add external datasets, probability calibration, class-imbalance handling and a deployment-grade security layer.

---

## 3. Likely viva questions and short answers

**Why Archerfish Optimization instead of PSO or GWO?**
AFO is a recent bio-inspired metaheuristic that is largely unexplored for diabetes feature selection. Its refraction-corrected aiming, peer competition and cooling jet dispersion provide a clear exploration/exploitation balance. The project implements it as a wrapper with a documented fitness function.

**Why did AFO select only four features?**
The objective includes a parsimony penalty, and the four-feature subset achieved the best trade-off between cross-validated F1/AUC and subset size. Glucose and BMI were selected by the whole final school; DPF and Pregnancies had strong consensus.

**Why is accuracy only about 75%?**
PIMA is a small, noisy clinical dataset with overlapping classes. Published PIMA results commonly fall in the mid-70s to low-80s depending on preprocessing and split, and a single 154-record test set has sampling uncertainty. No test data was used for tuning.

**Is data leakage possible?**
No. AFO only sees the 614 training records. Median imputation and scaling are fitted inside each training fold. The 154 test records are used once for the final metrics.

**Why Random Forest?**
It gave the best combination of accuracy, F1 and AUC among the AFO-optimized classifiers, handles non-linear feature interactions, and provides class probabilities. Hyperparameters were fixed before evaluation.

**Is this the exact AFO from a paper?**
The report does not include the exact equations. The implementation is explicitly documented as Archerfish-inspired, with its phases, equations, parameters and assumptions available on the AFO page and in the README. It is a genuine wrapper search, not a renamed random subset.

**What are the false negatives?**
24 diabetic records were classified as non-diabetic, giving 55.56% sensitivity. This is a limitation for screening. Future work should tune the decision threshold, apply class weighting or SMOTE, and report sensitivity/specificity trade-offs.

**Can a doctor use this?**
No. It is an academic prototype, not a medical device. It has not been externally validated and must not be used for diagnosis or treatment decisions.

**What would you improve next?**
External multi-centre validation, probability calibration, class-imbalance handling, a larger feature set, additional metaheuristics for comparison, and a secure deployment with authentication.

---

## 4. Recovery plan if something goes wrong

| Problem | Action |
|---|---|
| Browser shows "connection refused" | The server stopped. In the terminal: `source .venv/bin/activate && python app.py`. |
| Port 5055 busy | Stop the old process or run `PORT=5056 python app.py`, then open that port. |
| Dashboard shows zero records | The database was reset. Submit the two sample assessments, or restore `database/diabetes.db`. |
| Charts do not appear | Hard-refresh (`Cmd/Ctrl + Shift + R`). Confirm `static/js/vendor/chart.umd.min.js` exists. |
| AFO page missing / model error | `python train_models.py`, then restart the app. |
| "Re-optimize features" seems stuck | It takes about 20 seconds. Wait; do not click twice. |
| Everything fails | Open the seven PNG files in `screenshots/` and walk through those; the metrics and narrative are identical. |

Useful commands:

```bash
source .venv/bin/activate     # activate environment
python app.py                 # run the application
python train_models.py        # reproduce AFO + all models
python -m pytest -q           # 13 integration tests
rm database/diabetes.db       # clear demo history (schema is recreated)
```

---

## 5. Exact numbers to remember

- Dataset: **768 records**, **8 features**, 268 diabetic / 500 non-diabetic.
- Split: **614 train / 154 test**, stratified, seed 42.
- AFO: **8 → 4 features**, 22 archers, 12 iterations, 82 subsets, 15.46 s.
- AFO fitness: **0.274843 → 0.261424**; CV F1 **0.6439 → 0.6631**.
- Selected model: **Random Forest**, 4 features.
- Test: **74.68% accuracy**, **66.67% precision**, **55.56% recall**, **60.61% F1**, **0.8170 AUC**, **85.00% specificity**.
- Confusion matrix: **TN 85 / FP 15 / FN 24 / TP 30**.
- Conventional RF: **74.68% accuracy**, **59.79% F1**, **0.8128 AUC**.
- Demo sample result: **Diabetic, 57.5%, Moderate risk**.
