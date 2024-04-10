# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This section of the model card serves to answer basic questions regarding the model version, type and other details.

- **Person or organization developing model**
  The model was developed by me [Benjamin Knoepfle] during the course of the *Deploying a Scalable ML Pipeline in Production* course for the [*Machine Learning DevOps Engineer*](https://learn.udacity.com/nanodegrees/nd0821) nanodegree handed out by [*Udacity*](https://learn.udacity.com/).  
  Contact: benjamin.knoepfle@googlemail.com.
  A complete source repo can be found in my corresponding GitHub Repository [*Deploy-ML-Model-to-Cloud*](https://github.com/Benjamin-Knoepfle/Deploy-ML-Model-to-Cloud).

- **Model Date**
  The model was developed on April 2024

- **Model version**
  The model is in Version 1.0. 

- **Model type**:
  The used model is a *DecisionTreeClassifier* implemented within the *sklearn.tree* module.
  Used Scikit Learn version is 1.3.2

- **Paper or other resource for more information**:
  Further information on the development can be found within the [project requirenments](https://learn.udacity.com/nanodegrees/nd0821/parts/cd0582/lessons/ff6afa3c-d2a2-4343-bb14-d291a2b6d708/concepts/ff6afa3c-d2a2-4343-bb14-d291a2b6d708-project-rubric) needed to pass the course.

- **Citation details**:
  Feel free to cite it the way you want ;)

- **License**:
  This model developed by me runs under **no license**

- **Feedback on the model**
  Please feel free to reach out to me for questions or feedback given the model at hand.
  **Contact**: benjamin.knoepfle@googlemail.com

## Intended Use
This section allow readers to quickly grasp what the model
should and should not be used for, and why it was created.

- **Primary inteded uses**:
  This model is solely developed for educational purpose and part of the requirenment to pass the project. It has no relation to a real world scenario use-case.
  Model predicts if an income is below or above 50k dollars given some cencus data.

- **Primary intended users**:
  Intended users are the *reviewers* for the lecture and *me*.  

- **Out-of-scope uses**:
  Any use outside of the course project!

## Training Data
- **Datasets**: 80% of census dataset. The raw as well as the cleaned datasets can be found [here](https://github.com/Benjamin-Knoepfle/Deploy-ML-Model-to-Cloud/tree/master/starter/data).
- **Motivation**:
  Data is given by the course creators and its usage is mandatory.
- **Preprocessing**: 
  The raw dataset contains leading blanks to all column names and values. All leading blanks have been stripped.

## Evaluation Data
- **Datasets**: 20% of census dataset. The evaluation data is **non-overlapping** with the training data! The raw as well as the cleaned datasets can be found [here](https://github.com/Benjamin-Knoepfle/Deploy-ML-Model-to-Cloud/tree/master/starter/data).
- **Motivation**:
  Data is given by the course creators and its usage is mandatory.
- **Preprocessing**: 
  The raw dataset contains leading blanks to all column names and values. All leading blanks have been stripped.

## Metrics
The model was evaluated using [precision, recall](https://en.wikipedia.org/wiki/Precision_and_recall) and [f beta](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) score on the held out dataset described in Evaluation Data.
- **Precision**:    0.6140684410646388
- **Recall**:       0.6223506743737958
- **f beta**:       0.6181818181818182

## Ethical Considerations
The model can lead to discrimination if it relays protected features like sex, race or ethnics to create its predictions.
Further more the usage of a model like this could lead to privacy issues because it could give sensitive information about the financial status of a person. 

## Caveats and Recommendations
This model is solely developed for educational purpose and part of the requirenment to pass the project. It has no relation to a real world scenario use-case.
Please do not use the data or model for anything else as your own education ;)
