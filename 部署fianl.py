#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template, render_template_string
import os
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd


# In[2]:


import threading


# In[3]:


# 加载模型
xgb_model = xgb.Booster()
xgb_model.load_model('xgb_model.json')

# 加载目标编码器
with open('target_encoding1.pkl', 'rb') as f:
    target_encoding1 = pickle.load(f)

with open('target_encoding2.pkl', 'rb') as f:
    target_encoding2 = pickle.load(f)

# 加载特征名称
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)


# In[4]:


app = Flask(__name__)


# In[5]:


# 中英双语字典
language_texts = {
    'en': {
        'title': 'Predict Tool for Sludge Dewatering',
        'heading': 'Enter Data to Make Predictions',
        'catalyst_label': 'Catalyst',
        'radical_label': 'Radical Donor',
        'radical_concentration_label': 'Radical Donor C (mmol/L)',
        'catalyst_concentration_label': 'Catalyst C (mmol/L)',
        'ph_label': 'pH',
        'vs_ts_label': 'VS/TS',
        'sludge_water_content_label': 'Initial MC',
        'rpm_label': 'Mixing Speed (rpm)',
        'time_label': 'Mixing Time (min)',
        'predict_button': 'Predict',
        'result_heading': 'The Predicted Outcome Value',
        'back_button': 'Back to Home'
    },
    'zh': {
        'title': '污泥脱水预测工具',
        'heading': '输入数据以进行预测',
        'catalyst_label': '催化剂种类',
        'radical_label': '自由基种类',
        'radical_concentration_label': '自由基浓度 (mmol/L)',
        'catalyst_concentration_label': '催化剂浓度 (mmol/L)',
        'ph_label': 'pH',
        'vs_ts_label': 'VS/TS',
        'sludge_water_content_label': '原污泥含水率',
        'rpm_label': '转速 (rpm)',
        'time_label': '时间 (min)',
        'predict_button': '预测',
        'result_heading': '预测结果',
        'back_button': '返回主页'
    }
}


# In[6]:


def preprocess_input(data):
    # 编码类别特征
    catalyst_encoded = target_encoding1.get(data['催化剂种类'], 0)
    radical_encoded = target_encoding2.get(data['自由基种类'], 0)

    # 数值特征
    numerical_features = [
        float(data.get('自由基浓度mmol/L', 0)),
        float(data.get('催化剂浓度mmol/L', 0)),
        float(data.get('pH', 7.0)),
        float(data.get('VS/TS', 0)),
        float(data.get('原污泥含水率', 0)),
        float(data.get('转速rpm', 0)),
        float(data.get('时间/min', 0))
    ]

    # 合并特征并构建 DataFrame
    processed_features = [catalyst_encoded, radical_encoded] + numerical_features
    return pd.DataFrame([processed_features], columns=feature_names)


# In[8]:


# 初始化 Flask 应用
app = Flask(__name__)

# 模型和预处理相关内容
xgb_model = xgb.Booster()
xgb_model.load_model('xgb_model.json')

with open('target_encoding1.pkl', 'rb') as f:
    target_encoding1 = pickle.load(f)

with open('target_encoding2.pkl', 'rb') as f:
    target_encoding2 = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def preprocess_input(data):
    """
    数据预处理函数，将用户输入的数据转为模型可用的格式。
    """
    try:
        # 编码类别特征
        catalyst_encoded = target_encoding1.get(data['催化剂种类'], 0)
        radical_encoded = target_encoding2.get(data['自由基种类'], 0)

        # 数值特征
        numerical_features = [
            float(data.get('自由基浓度mmol/L', 0)),
            float(data.get('催化剂浓度mmol/L', 0)),
            float(data.get('pH', 7.0)),
            float(data.get('VS/TS', 0)),
            float(data.get('原污泥含水率', 0)),
            float(data.get('转速rpm', 0)),
            float(data.get('时间/min', 0))
        ]

        # 合并所有特征
        processed_features = [catalyst_encoded, radical_encoded] + numerical_features

        # 创建 DataFrame 并确保特征名称对齐
        return pd.DataFrame([processed_features], columns=feature_names)
    except Exception as e:
        raise ValueError(f"预处理错误: {str(e)}")

# 自由基种类和催化剂种类的映射规则
radical_merge_map = {
    'SPS过硫酸盐': 'SPS'
}
catalyst_merge_map = {
    'Fe Ⅱ': 'FeⅡ'
}

# 原始下拉菜单选项
catalyst_types_raw = ['FeⅡ', 'ZVI', 'PMS', 'Fe3O4', 'Al', 'FeⅢ', 'Fe Ⅱ']
radical_types_raw = ['SPS', 'H2O2', 'KMnO4', 'SPC', 'O3', 'PMS', 'Fe Ⅵ', 'KHSO5', 'sodium sulphite', 'SPS过硫酸盐']

# 应用映射规则并去重
catalyst_types = list(set(catalyst_merge_map.get(c, c) for c in catalyst_types_raw))
radical_types = list(set(radical_merge_map.get(r, r) for r in radical_types_raw))

@app.route('/')
def home():
    """
    主页：提供语言选择和输入表单。
    """
    # 获取语言参数（默认为中文 'zh'）
    lang = request.args.get('lang', 'zh')
    texts = language_texts.get(lang, language_texts['zh'])  # 根据语言获取文本

    # 动态生成下拉菜单选项
    catalyst_options = ''.join([f'<option value="{c}">{c}</option>' for c in catalyst_types])
    radical_options = ''.join([f'<option value="{r}">{r}</option>' for r in radical_types])

    return render_template_string(f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>{texts['title']}</title>
    </head>
    <body>
        <h1>{texts['heading']}</h1>
        <form action="/predict?lang={lang}" method="post">
            <label>{texts['catalyst_label']}:</label>
            <select name="catalyst">
                {catalyst_options}
            </select><br>
            
            <label>{texts['radical_label']}:</label>
            <select name="radical">
                {radical_options}
            </select><br>
            
            <label>{texts['radical_concentration_label']}:</label>
            <input name="radical_concentration" type="number" step="0.01"><br>
            <label>{texts['catalyst_concentration_label']}:</label>
            <input name="catalyst_concentration" type="number" step="0.01"><br>
            <label>{texts['ph_label']}:</label>
            <input name="pH" type="number" step="0.01" min="0" max="14"><br>
            <label>{texts['vs_ts_label']}:</label>
            <input name="vs_ts" type="number" step="0.01"><br>
            <label>{texts['sludge_water_content_label']}:</label>
            <input name="sludge_water_content" type="number" step="0.01" min="1" max="100"><br>
            <label>{texts['rpm_label']}:</label>
            <input name="rpm" type="number" step="1"><br>
            <label>{texts['time_label']}:</label>
            <input name="time" type="number" step="1"><br>
            <button type="submit">{texts['predict_button']}</button>
        </form>

        <p>
            <a href="/?lang=zh">中文</a> | <a href="/?lang=en">English</a>
        </p>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取语言参数（默认为中文 'zh'）
        lang = request.args.get('lang', 'zh')
        texts = language_texts.get(lang, language_texts['zh'])

        # 收集用户输入数据
        data = {
            "催化剂种类": catalyst_merge_map.get(request.form['catalyst'], request.form['catalyst']),
            "自由基种类": radical_merge_map.get(request.form['radical'], request.form['radical']),
            "自由基浓度mmol/L": float(request.form['radical_concentration']),
            "催化剂浓度mmol/L": float(request.form['catalyst_concentration']),
            "pH": float(request.form['pH']),
            "VS/TS": float(request.form['vs_ts']),
            "原污泥含水率": float(request.form['sludge_water_content']),
            "转速rpm": float(request.form['rpm']),
            "时间/min": float(request.form['time']),
        }

        # 数据预处理
        features = preprocess_input(data)

        # 转换为 DMatrix
        dmatrix = xgb.DMatrix(features)

        # 进行预测
        prediction = xgb_model.predict(dmatrix)

        # 返回预测结果页面
        return render_template_string(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>{texts['result_heading']}</title>
        </head>
        <body>
            <h1>{texts['result_heading']}</h1>
            <p>{texts['result_heading']}: <strong>{{{{ prediction }}}}</strong></p>
            <a href="/?lang={lang}">{texts['back_button']}</a>
        </body>
        </html>
        ''', prediction=prediction[0])
    except Exception as e:
        return f"发生错误: {str(e)}"
        
# 启动 Flask 应用
def run_flask():
    app.run(debug=True, use_reloader=False, port=5000)

# 使用线程启动 Flask
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True  # 设置为守护线程
flask_thread.start()

print("Flask 应用已启动，访问 http://127.0.0.1:5000")


# In[10]:


get_ipython().system('jupyter nbconvert --to script 部署fianl.ipynb')


# In[ ]:




