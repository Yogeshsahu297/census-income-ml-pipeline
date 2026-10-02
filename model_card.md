# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed by the project author as part of the "Deploying a Scalable ML Pipeline in Production" project. It is a scikit-learn `RandomForestClassifier` with 100 trees, a maximum depth of 20, a minimum of 5 samples per leaf and a fixed random seed of 42. The eight categorical features are one-hot encoded with a `OneHotEncoder` (unknown categories are ignored at inference time) and the label is binarized with a `LabelBinarizer`; the six continuous features are passed through unscaled because tree ensembles are insensitive to feature scale. The fitted model, encoder and label binarizer are stored in `model/` as `model.pkl`, `encoder.pkl` and `lb.pkl`. The model was developed with Python 3.12.3, scikit-learn 1.8.0 and pandas 3.0.2, and it must be loaded with the same scikit-learn version.

## Intended Use

The model predicts whether a person's annual income exceeds $50,000 from demographic and employment attributes in the US Census "Adult" data. It is intended for educational purposes: to demonstrate a reproducible machine learning pipeline that is tested, continuously integrated and deployed behind a REST API. It is not intended for any real decision about individuals, such as credit, hiring, housing or insurance decisions.

## Training Data

The model was trained on the Census Income (Adult) dataset extracted from the 1994 US Census database, provided as `data/census.csv`. The file contains 32,561 rows and 14 input features plus the `salary` label (`<=50K` or `>50K`). About 24.1% of the rows belong to the positive class (`>50K`). The data was split once into 80% training data (26,048 rows) and 20% test data (6,513 rows) using a stratified split with a fixed seed, so the class balance is the same in both parts. Missing values are encoded in the source data as `?` (4,262 occurrences, mostly in `workclass`, `occupation` and `native-country`); they were kept and are treated as their own category.

## Evaluation Data

The evaluation data is the held-out 20% test split (6,513 rows). It was transformed with the encoder and label binarizer that were fitted on the training split only, so no information from the test split leaked into preprocessing.

## Metrics

The model is evaluated with precision, recall and the F1 score (F-beta with beta = 1) for the positive class `>50K`. On the held-out test set the model achieves a precision of 0.7919, a recall of 0.5995 and an F1 score of 0.6824. In other words, when the model predicts that someone earns more than $50K it is right about 79% of the time, but it finds only about 60% of the people who actually do.

The same metrics were also computed on slices of the test data for every unique value of each categorical feature; the full results are in `slice_output.txt`. Performance varies noticeably between slices. For example, by sex the F1 score is 0.6598 for female and 0.6861 for male records (recall is 0.5265 versus 0.6130), and by race it ranges from 0.4615 for Amer-Indian-Eskimo (73 test rows) to 0.6879 for White (5,533 test rows). Slices with only a handful of rows, such as `race=Other` or `workclass=Never-worked`, produce metrics that are statistically meaningless and should not be interpreted.

## Ethical Considerations

The dataset contains sensitive attributes, including race, sex, marital status and native country, and it reflects the social structure of the United States in 1994, including historical income inequalities. A model trained on it can reproduce or amplify those inequalities. The slice results show that recall is lower for women and for several racial groups than for the overall population, which means the model would systematically under-identify high earners in those groups. The data is also heavily imbalanced in representation (about 85% White and 67% male), so small groups are evaluated on very few examples. The model must not be used to make decisions that affect people's opportunities. The dataset contains personal-style records, although it is a public, anonymized research dataset.

## Caveats and Recommendations

The data is more than 30 years old, so both income levels (the $50K threshold is not inflation adjusted) and the relationship between the features and income have changed; predictions on present-day data will not be reliable. The model only knows the categories seen in training, and an unseen category is silently encoded as all zeros. The `fnlgt` sampling weight is used as an ordinary feature, which is a modeling simplification. Recall for the positive class is only about 60%, so the decision threshold could be tuned if recall matters more than precision. Before any real-world use, the model should be retrained on current data, audited for fairness across demographic groups with larger and better-balanced samples, and evaluated with cross-validation or confidence intervals to quantify uncertainty.
