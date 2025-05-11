import pandas as pd
import joblib

def load_models():
    """Load trained model and encoders with error handling"""
    try:
        model = joblib.load('model_naive_bayes.pkl')
        encoders = joblib.load('feature_encoders.pkl')
        label_encoder = joblib.load('target_encoder.pkl')
        
        # Validasi model dan encoder
        if not all(col in encoders for col in ['umur', 'tinggi', 'jumlah_daun', 'kondisi_daun']):
            raise ValueError("Encoders tidak lengkap!")
            
        return model, encoders, label_encoder
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def validate_input(prompt, valid_options):
    """Validate user input with case-insensitive check"""
    while True:
        user_input = input(prompt).strip().lower()
        for option in valid_options:
            if user_input == option.lower():
                return option.capitalize()
        print(f"Error: Hanya menerima {', '.join(valid_options)}")

def generate_explanation(model, kelas, feature_values, feature_names, label_encoder):
    explanations = []
    try:
        feature_impacts = {
            name: value for name, value in zip(feature_names, feature_values)
        }

        if feature_impacts['umur'].lower() == 'ya':
            explanations.append("Pilihan umur panen cepat cocok untuk varietas ini")
        else:
            explanations.append("Varietas ini baik untuk pertumbuhan jangka panjang")

        if feature_impacts['tinggi'].lower() == 'ya':
            explanations.append("Tanaman tinggi cocok untuk lahan luas")
        else:
            explanations.append("Tanaman pendek ideal untuk lahan terbatas")

        if feature_impacts['jumlah_daun'].lower() == 'banyak':
            explanations.append("Daun rimbun meningkatkan produktivitas")
        else:
            explanations.append("Daun sedikit memudahkan perawatan")

        if feature_impacts['kondisi_daun'].lower() == 'baik':
            explanations.append("Kondisi daun optimal untuk hasil terbaik")
        else:
            explanations.append("Varietas ini toleran terhadap kondisi daun kurang ideal")

        explanations.append("Rekomendasi akhir berdasarkan kombinasi karakteristik yang paling sesuai.")
    except Exception as e:
        explanations.append("[System] Penjelasan teknis tidak tersedia")
    
    return explanations


def predict_cabai(model, encoders, label_encoder, umur, tinggi, jumlah_daun, kondisi_daun):
    """Safe prediction function with comprehensive checks"""
    try:
        # Validasi input
        input_features = {
            'umur': umur,
            'tinggi': tinggi,
            'jumlah_daun': jumlah_daun,
            'kondisi_daun': kondisi_daun
        }
        
        for col, value in input_features.items():
            if value.lower() not in [x.lower() for x in encoders[col].classes_]:
                raise ValueError(f"Nilai '{value}' tidak valid untuk {col}. Pilihan: {list(encoders[col].classes_)}")
        
        # Encode input
        input_encoded = [
            encoders['umur'].transform([umur])[0],
            encoders['tinggi'].transform([tinggi])[0],
            encoders['jumlah_daun'].transform([jumlah_daun])[0],
            encoders['kondisi_daun'].transform([kondisi_daun])[0]
        ]
        
        # Prediksi
        input_df = pd.DataFrame([input_encoded], columns=['umur', 'tinggi', 'jumlah_daun', 'kondisi_daun'])
        proba = model.predict_proba(input_df)[0]
        kelas = label_encoder.inverse_transform([proba.argmax()])[0]
        
        # Generate explanation
        explanations = generate_explanation(
            model, kelas,
            [umur, tinggi, jumlah_daun, kondisi_daun],
            ['umur', 'tinggi', 'jumlah_daun', 'kondisi_daun'],
            label_encoder
        )
        
        return {
            'Rekomendasi': kelas,
            'Probabilitas': f"{proba.max()*100:.1f}%",
            'Detail': {k: f"{v*100:.1f}%" for k, v in zip(label_encoder.classes_, proba)},
            'Penjelasan': explanations
        }
        
    except Exception as e:
        return {"Error": str(e)}

def main():
    model, encoders, label_encoder = load_models()
    if None in [model, encoders, label_encoder]:
        print("Gagal memuat model. Pastikan file model ada dan valid.")
        return
    
    print("\n" + "="*40)
    print("SISTEM REKOMENDASI BIBIT CABAI UNGGUL")
    print("="*40)
    
    # Get user input
    umur = validate_input("- Umur (Ya/Tidak): ", ['Ya', 'Tidak'])
    tinggi = validate_input("- Tinggi (Ya/Tidak): ", ['Ya', 'Tidak'])
    jumlah_daun = validate_input("- Jumlah Daun (Banyak/Sedikit): ", ['Banyak', 'Sedikit'])
    kondisi_daun = validate_input("- Kondisi Daun (Baik/Buruk): ", ['Baik', 'Buruk'])
    
    # Get prediction
    hasil = predict_cabai(model, encoders, label_encoder, umur, tinggi, jumlah_daun, kondisi_daun)
    
    # Display results
    print("\n🔍 HASIL REKOMENDASI:")
    if 'Error' in hasil:
        print(f"Error: {hasil['Error']}")
        print("\n⚠️ Solusi:")
        print("- Pastikan model telah dilatih dengan data yang benar")
        print("- Cek konsistensi nilai input dengan data training")
    else:
        print(f"Bibit terbaik: {hasil['Rekomendasi']} ({hasil['Probabilitas']})")
        
        print("\n📊 Analisis Keputusan:")
        for explanation in hasil['Penjelasan']:
            print(f"- {explanation}")
        
        print("\nProbabilitas Lengkap:")
        for jenis, prob in sorted(hasil['Detail'].items(), 
                                key=lambda x: float(x[1].replace('%', '')), 
                                reverse=True):
            print(f"- {jenis}: {prob}")

if __name__ == "__main__":
    main()
    