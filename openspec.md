# Spam Analytics Toolkit — OpenSpec (v0.1)

## Project
- id: `spam-analytics`
- name: Spam Analytics Toolkit
- description: Research-oriented workflow for exploring, modeling, and presenting the SMS Spam dataset. Includes preprocessing stages, evaluation metrics, CLI automation, and a Streamlit dashboard.

## Datasets
### `sms_spam_v1`
- path: `dataset.csv`
- format: `csv`
- has_header: `false`
- usage: train, evaluate, visualize

**Columns**

| name  | type         | description                                    |
|-------|--------------|--------------------------------------------------|
| label | categorical  | Message label (`ham` or `spam`).                |
| text  | string       | Raw SMS message content.                        |

## Artifacts

| id                  | path                                     | description                                                                       |
|---------------------|------------------------------------------|-----------------------------------------------------------------------------------|
| model_bundle        | `artifacts/model.joblib`                 | Serialized scikit-learn pipeline containing text vectorizer and classifier.       |
| preprocessing_cache | `artifacts/preprocessing_steps.parquet`  | Snapshot of preprocessing steps for each message.                                 |
| metrics_report      | `reports/metrics.json`                   | Evaluation metrics (accuracy, precision, recall, f1, roc_auc).                    |
| figures             | `reports/figures/`                       | Generated plots for dataset insights and model diagnostics.                        |

## Workflows

### CLI Training
- id: `cli_training`
- name: Train via CLI
- entrypoint:

```bash
spam-analytics model train --config spam_analytics/config.yaml
```

- summary: Load dataset, run preprocessing pipeline, train classifier, persist artifacts, and log metrics.
- inputs: `sms_spam_v1`
- outputs: `model_bundle`, `preprocessing_cache`, `metrics_report`

**Steps**
- `load_dataset`: Read dataset.csv and coerce to canonical schema.
- `preprocess_text`: Run normalization, tokenization, stopword removal, stemming, and feature generation.
- `vectorize`: Fit TF-IDF vectorizer with char and word n-grams.
- `train_model`: Fit logistic regression classifier with class balancing.
- `evaluate_model`: Produce cross-validation and hold-out metrics; dump to reports/metrics.json.
- `persist_artifacts`: Save trained pipeline and preprocessing snapshot to artifacts/.

---

### Generate visual summaries
- id: `cli_visualize`
- name: Generate visual summaries
- entrypoint:

```bash
spam-analytics visualize report
```

- summary: Create plots that explore class balance, message lengths, token frequencies, and evaluation diagnostics.
- inputs: `sms_spam_v1`, `metrics_report`
- outputs: `figures`

---

### Streamlit Dashboard
- id: `streamlit_dashboard`
- name: Streamlit Dashboard
- entrypoint:

```bash
streamlit run streamlit_app.py
```

- summary: Interactive dashboard for data inspection, preprocessing previews, model metrics, and inference.
- inputs: `sms_spam_v1`, `preprocessing_cache`, `metrics_report`, `model_bundle`

