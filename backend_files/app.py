import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / 'superkart_model.joblib')
meta = json.loads((ROOT / 'metadata.json').read_text())
superkart_api = Flask(__name__)
superkart_api.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

def validate(frame):
    missing = sorted(set(meta['features']) - set(frame.columns))
    if missing: raise ValueError('Missing columns: ' + ', '.join(missing))
    if len(frame) == 0 or len(frame) > 10000: raise ValueError('Provide 1 to 10000 rows.')
    frame = frame[meta['features']].copy()
    for col in meta['numeric']:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col]).all(): raise ValueError(col + ' must be finite and nonmissing.')
        if (frame[col] < 0).any(): raise ValueError(col + ' must be nonnegative.')
    if (frame['Product_Allocated_Area'] > 1).any(): raise ValueError('Allocated area must be between 0 and 1.')
    for col in meta['categorical']:
        if frame[col].isna().any(): raise ValueError(col + ' must not be missing.')
        frame[col] = frame[col].astype(str).str.strip()
        if frame[col].eq('').any(): raise ValueError(col + ' must not be empty.')
    frame['Product_Sugar_Content'] = frame['Product_Sugar_Content'].replace({'reg':'Regular'})
    notices = {c:sorted(set(frame[c])-set(meta['categories'][c])) for c in meta['categorical']}
    return frame, {k:v for k,v in notices.items() if v}

def infer(frame):
    frame, notices = validate(frame)
    predictions = model.predict(frame)
    if not np.isfinite(predictions).all(): raise RuntimeError('Nonfinite prediction.')
    return predictions.tolist(), notices

@superkart_api.get('/')
@superkart_api.get('/health')
def health(): return jsonify(status='ok', model=meta['selected_model'])

@superkart_api.post('/v1/predict')
def predict():
    payload = request.get_json()
    if not isinstance(payload,dict): raise ValueError('Expected one JSON object.')
    predictions, notices = infer(pd.DataFrame([payload]))
    return jsonify(prediction=predictions[0], unknown_categories=notices)

@superkart_api.post('/v1/predictbatch')
def predict_batch():
    if 'file' not in request.files: raise ValueError('Upload a CSV using multipart field file.')
    try: frame = pd.read_csv(request.files['file'])
    except (pd.errors.ParserError,pd.errors.EmptyDataError,UnicodeDecodeError) as exc:
        raise ValueError('Invalid CSV.') from exc
    predictions, notices = infer(frame)
    return jsonify(predictions=predictions, row_count=len(predictions), unknown_categories=notices)

@superkart_api.errorhandler(ValueError)
@superkart_api.errorhandler(TypeError)
@superkart_api.errorhandler(BadRequest)
def invalid(exc): return jsonify(error=str(exc)), 400

@superkart_api.errorhandler(RequestEntityTooLarge)
def too_large(exc): return jsonify(error='Upload exceeds 5 MB.'), 413

if __name__ == '__main__':
    superkart_api.run(host='0.0.0.0',port=7860,debug=False)
