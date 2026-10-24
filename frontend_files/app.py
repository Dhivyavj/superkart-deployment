import os, json
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
BASE = os.environ.get('BACKEND_URL','http://backend:7860').rstrip('/')
meta = json.loads((Path(__file__).parent/'metadata.json').read_text())
st.set_page_config(page_title='SuperKart Revenue Predictor',layout='wide')
st.title('SuperKart Revenue Predictor')
st.caption('Revenue estimates based on historical product–store relationships; not a seasonality-aware quarterly forecast.')

def show_error(exc):
    st.error('API request failed: ' + str(exc))

single, batch = st.tabs(['Single prediction','Batch CSV'])
with single:
    with st.form('prediction'):
        values = {}
        for col in meta['features']:
            if col in meta['numeric']:
                default = {'Product_Weight':12.66,'Product_Allocated_Area':.027,'Product_MRP':117.08,'Store_Age_Years':16.0}[col]
                values[col] = st.number_input(col,min_value=0.0,value=default,
                    max_value=1.0 if col=='Product_Allocated_Area' else None,format='%.3f')
            else:
                values[col] = st.selectbox(col,meta['categories'][col])
        submit = st.form_submit_button('Predict revenue')
    if submit:
        try:
            response = requests.post(BASE+'/v1/predict',json=values,timeout=60)
            response.raise_for_status(); result=response.json()
            st.metric('Predicted revenue (dataset currency units)',f"{result['prediction']:,.2f}")
            if result['unknown_categories']: st.warning(result['unknown_categories'])
        except (requests.RequestException,ValueError,KeyError) as exc: show_error(exc)
with batch:
    st.write('Required columns:',', '.join(meta['features']))
    uploaded = st.file_uploader('Upload engineered-feature CSV',type=['csv'])
    if uploaded is not None and st.button('Predict batch'):
        try:
            contents=uploaded.getvalue()
            response=requests.post(BASE+'/v1/predictbatch',files={'file':('batch.csv',contents,'text/csv')},timeout=120)
            response.raise_for_status(); result=response.json()
            output=pd.read_csv(__import__('io').BytesIO(contents))
            output['Predicted_Sales']=result['predictions']
            st.dataframe(output)
            if result['unknown_categories']: st.warning('Unseen categories were encoded without crashing: '+str(result['unknown_categories']))
            st.download_button('Download predictions',output.to_csv(index=False),'superkart_predictions.csv','text/csv')
        except (requests.RequestException,ValueError,KeyError) as exc: show_error(exc)
