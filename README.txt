# Employee Attrition Prediction
## Random Forest Model — Test Accuracy: ~87%

### Setup & Run
```
pip install -r requirements.txt
python app.py
```
Open: http://127.0.0.1:5000

### Retrain Model (optional)
```
# Place WA_Fn-UseC_-HR-Employee-Attrition.csv in same folder
python train_model.py
```

### Why ~87% accuracy (not 100%)?
- max_depth=10 prevents memorizing training data
- Tested on unseen 20% test split
- 5-Fold Cross Validation confirms generalization
- This is the REAL, honest accuracy
