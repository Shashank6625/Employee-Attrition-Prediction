from flask import Flask, render_template, request, jsonify
import pickle, json, numpy as np, os, io
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__,
             template_folder=os.path.join(BASE,'templates'),
             static_folder  =os.path.join(BASE,'static'))

model      = pickle.load(open(os.path.join(BASE,'model.pkl'),'rb'))
le_dict    = json.load(open(os.path.join(BASE,'label_encoders.json')))
feat_cols  = json.load(open(os.path.join(BASE,'feature_cols.json')))
feat_imp   = json.load(open(os.path.join(BASE,'feature_importances.json')))
emp_lookup = json.load(open(os.path.join(BASE,'employee_lookup.json')))
config     = json.load(open(os.path.join(BASE,'config.json')))

THRESHOLD = config['threshold']
MAX_PROB  = config.get('max_prob', 0.85)
ACCURACY  = config['accuracy']
CV_ACC    = config.get('cv_mean_accuracy', 0)
CV_STD    = config.get('cv_std', 0)

CAT_COLS = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']

def scale_risk(prob):
    return round(float(np.clip(prob / MAX_PROB * 100, 0, 100)), 1)

def risk_level(risk):
    if risk >= 75: return 'CRITICAL'
    if risk >= 50: return 'HIGH'
    if risk >= 25: return 'MEDIUM'
    return 'LOW'

def get_factors(data):
    factors = []
    if str(data.get('OverTime',''))                  == 'Yes':               factors.append('Frequent Overtime')
    if float(data.get('JobSatisfaction',3))          <= 2:                   factors.append('Low Job Satisfaction')
    if float(data.get('WorkLifeBalance',3))          <= 2:                   factors.append('Poor Work-Life Balance')
    if float(data.get('YearsAtCompany',5))           <  3:                   factors.append('Short Tenure')
    if float(data.get('MonthlyIncome',5000))         <  3000:                factors.append('Below Avg Income')
    if float(data.get('EnvironmentSatisfaction',3))  <= 2:                   factors.append('Low Env Satisfaction')
    if float(data.get('YearsSinceLastPromotion',2))  >  5:                   factors.append('No Recent Promotion')
    if str(data.get('BusinessTravel',''))            == 'Travel_Frequently': factors.append('Frequent Travel')
    if float(data.get('NumCompaniesWorked',1))       >  5:                   factors.append('High Job-Hopping')
    if float(data.get('RelationshipSatisfaction',3)) <= 2:                   factors.append('Low Relationship Sat')
    if float(data.get('PerformanceRating',3))        <= 2:                   factors.append('Low Performance Rating')
    return factors

def get_retention_actions(factors):
    action_map = {
        'Frequent Overtime'     : '⏰ Reduce overtime — assign backup support',
        'Low Job Satisfaction'  : '😊 Schedule 1:1 feedback session',
        'Poor Work-Life Balance': '🏠 Offer flexible work-from-home options',
        'Short Tenure'          : '🤝 Assign a senior mentor',
        'Below Avg Income'      : '💰 Review compensation — offer hike or bonus',
        'Low Env Satisfaction'  : '🏢 Address workplace concerns',
        'No Recent Promotion'   : '📈 Create clear promotion roadmap with 6-month goals',
        'Frequent Travel'       : '✈️ Limit travel frequency',
        'High Job-Hopping'      : '🔒 Offer long-term incentives like ESOP',
        'Low Relationship Sat'  : '👥 Facilitate team-building activities',
        'Low Performance Rating': '📚 Enroll in upskilling program',
    }
    actions = [action_map[f] for f in factors if f in action_map]
    if not actions:
        actions.append('✅ Employee is stable — continue regular engagement check-ins')
    return actions

def encode_and_predict(data):
    predict_data = dict(data)
    if 'Department_ibm' in predict_data: predict_data['Department'] = predict_data['Department_ibm']
    if 'JobRole_ibm'    in predict_data: predict_data['JobRole']    = predict_data['JobRole_ibm']

    for col in CAT_COLS:
        val = str(predict_data.get(col,''))
        predict_data[col+'_enc'] = le_dict[col].get(val, 0)

    feats = [float(predict_data.get(f,0)) for f in feat_cols]
    X     = np.array(feats).reshape(1,-1)
    proba = model.predict_proba(X)[0]
    prob  = float(proba[1])
    risk  = scale_risk(prob)
    pred  = 1 if prob >= THRESHOLD else 0
    level = risk_level(risk)
    factors = get_factors(data)
    actions = get_retention_actions(factors)

    return {
        'prediction'         : pred,
        'attrition_risk'     : risk,
        'stay_probability'   : round(100-risk, 1),
        'risk_level'         : level,
        'key_factors'        : factors,
        'retention_actions'  : actions,
        'feature_importances': feat_imp,
    }

@app.route('/')
def index():
    return render_template('index.html', accuracy=ACCURACY, cv_acc=CV_ACC, cv_std=CV_STD)

@app.route('/api/employees')
def get_employees():
    employees = [{'id':k,'name':v.get('Name',f'EMP-{k}'),'dept':v.get('Department','—'),
                  'role':v.get('JobRole','—'),
                  'label':f"EMP-{k} — {v.get('Department','?')} | {v.get('JobRole','?')}"}
                 for k,v in emp_lookup.items()]
    employees.sort(key=lambda x: int(x['id']))
    return jsonify(employees)

@app.route('/api/employee/<emp_id>')
def get_employee(emp_id):
    emp = emp_lookup.get(str(emp_id))
    if not emp: return jsonify({'error':'Employee not found'}), 404
    return jsonify(emp)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        return jsonify(encode_and_predict(request.get_json()))
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        file = request.files.get('file')
        if not file: return jsonify({'error':'No file uploaded'}), 400
        raw = file.read()
        try:    content = raw.decode('utf-8-sig')
        except: content = raw.decode('latin-1')
        df = pd.read_csv(io.StringIO(content))
        if df.empty: return jsonify({'error':'CSV empty'}), 400
        df.columns = df.columns.str.strip()

        emp_ids     = df.get('EmployeeNumber',pd.Series(['—']*len(df))).astype(str).str.strip()
        departments = df.get('Department',    pd.Series(['—']*len(df))).astype(str).str.strip()
        job_roles   = df.get('JobRole',       pd.Series(['—']*len(df))).astype(str).str.strip()
        names       = df.get('Name',          pd.Series(['']*len(df))).astype(str).str.strip()

        df_enc = df.copy()
        for col in CAT_COLS:
            df_enc[col+'_enc'] = df_enc[col].astype(str).map(le_dict[col]).fillna(0).astype(int) if col in df_enc.columns else 0

        X = np.zeros((len(df_enc),len(feat_cols)),dtype=float)
        for j,col in enumerate(feat_cols):
            if col in df_enc.columns:
                X[:,j] = pd.to_numeric(df_enc[col],errors='coerce').fillna(0).values

        probs = model.predict_proba(X)[:,1]
        results = []
        for i,prob in enumerate(probs):
            try:
                risk  = scale_risk(prob)
                facts = get_factors({c: df.iloc[i].get(c,'') for c in df.columns})
                results.append({
                    'emp_id':emp_ids.iloc[i],'name':names.iloc[i] if names.iloc[i]!='nan' else '',
                    'department':departments.iloc[i],'job_role':job_roles.iloc[i],
                    'risk':risk,'level':risk_level(risk),'prediction':1 if prob>=THRESHOLD else 0,
                    'factors':facts,'actions':get_retention_actions(facts)
                })
            except: pass

        if not results: return jsonify({'error':'No valid rows'}), 400
        results.sort(key=lambda x:x['risk'],reverse=True)
        return jsonify({'results':results,'total':len(results)})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

if __name__ == '__main__':
    print(f"  Model Accuracy : {ACCURACY}%")
    print(f"  CV Accuracy    : {CV_ACC}% +/- {CV_STD}%")
    print("  Open: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
