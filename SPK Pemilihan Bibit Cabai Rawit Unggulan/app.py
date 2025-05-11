from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia123'

# Load model dan encoder
def load_models():
    try:
        model = joblib.load(Path('pkl/model_naive_bayes.pkl'))
        feature_encoder = joblib.load(Path('pkl/feature_encoders.pkl'))
        target_encoder = joblib.load(Path('pkl/target_encoder.pkl'))
        return model, feature_encoder, target_encoder
    except Exception as e:
        print(f"Error saat memuat model/encoder: {str(e)}")
        return None, None, None

model, feature_encoder, target_encoder = load_models()

# Fungsi penjelasan keputusan (bahasa mudah dipahami petani)
def generate_explanation(feature_values, feature_names):
    explanations = []
    try:
        feature_impacts = {
            name: value for name, value in zip(feature_names, feature_values)
        }

        if feature_impacts['umur'].lower() == 'ya':
            explanations.append("Bibit ini cocok untuk panen cepat, bisa dipanen dalam waktu singkat.")
        else:
            explanations.append("Bibit ini lebih baik untuk jangka panjang, hasil lebih besar tapi butuh waktu lebih lama.")

        if feature_impacts['tinggi'].lower() == 'ya':
            explanations.append("Tanaman ini tumbuh tinggi, cocok ditanam di lahan luas dan terbuka.")
        else:
            explanations.append("Tanaman pendek, cocok untuk lahan sempit atau dekat rumah.")

        if feature_impacts['jumlah_daun'].lower() == 'banyak':
            explanations.append("Daunnya banyak, bisa mendukung pertumbuhan buah lebih banyak.")
        else:
            explanations.append("Daunnya sedikit, mudah dirawat dan tidak terlalu lebat.")

        if feature_impacts['kondisi_daun'].lower() == 'baik':
            explanations.append("Daun dalam kondisi bagus, tandanya tanaman sehat dan kuat.")
        else:
            explanations.append("Meski daun kurang bagus, bibit ini masih bisa tumbuh dengan baik.")
            
    except Exception as e:
        explanations.append("[System] Penjelasan teknis tidak tersedia.")
    
    return explanations

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = {
            'umur': request.form.get('umur'),
            'tinggi': request.form.get('tinggi'),
            'jumlah_daun': request.form.get('jumlah_daun'),
            'kondisi_daun': request.form.get('kondisi_daun')
        }

        try:
            # Encode input
            input_encoded = [
                feature_encoder['umur'].transform([data['umur']])[0],
                feature_encoder['tinggi'].transform([data['tinggi']])[0],
                feature_encoder['jumlah_daun'].transform([data['jumlah_daun']])[0],
                feature_encoder['kondisi_daun'].transform([data['kondisi_daun']])[0]
            ]
            
            input_df = pd.DataFrame([input_encoded], 
                                  columns=['umur', 'tinggi', 'jumlah_daun', 'kondisi_daun'])
            
            # Prediksi
            proba = model.predict_proba(input_df)[0]
            prediction = target_encoder.inverse_transform([proba.argmax()])[0]

            # Penjelasan
            explanations = generate_explanation(
                [data['umur'], data['tinggi'], data['jumlah_daun'], data['kondisi_daun']],
                ['umur', 'tinggi', 'jumlah_daun', 'kondisi_daun']
            )
            
            return render_template('result.html',
                               prediction=prediction,
                               probability=f"{proba.max()*100:.1f}%",
                               details=dict(zip(target_encoder.classes_, [f"{p*100:.1f}%" for p in proba])),
                               explanations=explanations)
        
        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
