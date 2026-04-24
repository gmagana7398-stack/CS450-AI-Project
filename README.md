# CS450 Project - AI Attrition Prediction

This is our project for predicting if employees will quit or get burned out in AI jobs. It uses a logistic regression model and then uses OpenAI to explain the results.

HOW TO SETUP
------------
Since the .venv folder is ignored, you have to make your own when you first download the code. A virtual environment (venv) is basically a private sandbox for this project. It keeps all the libraries we use separate so they don't interfere with other Python stuff on your computer.

For Mac:
1. cd CS450FinalProject
2. python3 -m venv .venv
3. source .venv/bin/activate

For Windows:
1. cd CS450FinalProject
2. python -m venv .venv
3. .venv\Scripts\activate

INSTALLING LIBRARIES
--------------------
Once you have your venv on, run this command to get all the libraries:
pip install -r requirements.txt

If you install something new, remember to update the list:
pip freeze > requirements.txt
Then git add and push the requirements.txt file.

HOW TO RUN THE CODE
-------------------
1. Make a file called .env in the folder.
2. Inside it, put: OPENAI_API_KEY=your_key_here
3. Run "python Predict.py" to start the survey.
4. Run "python Explainer.py" just to test a single person.

MODEL STATS
-----------
- Model used: Logistic Regression
- F1 Score: 0.6103
- ROC-AUC: 0.8937
