import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ========================
# 1. Load Model and Encoders
# ========================
def load_models():
    """Load trained model and encoders"""
    try:
        model = joblib.load('model_naive_bayes.pkl')
        encoders = joblib.load('feature_encoders.pkl')  # Contains all feature encoders
        label_encoder = joblib.load('target_encoder.pkl')  # Separate encoder for target
        return model, encoders, label_encoder
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please make sure you have trained the model first.")
        return None, None, None

# ========================
# 2. Input Validation
# ========================
def validate_input(prompt, valid_options):
    """Validate user input against allowed options"""
    while True:
        user_input = input(prompt).strip().capitalize()
        if user_input in valid_options:
            return user_input
        print(f"Error: Only accepts {', '.join(valid_options)}")

# ========================
# 3. Prediction Function
# ========================
def predict_cabai(model, encoders, label_encoder, umur, tinggi, jumlah_daun, kondisi_daun):
    """Predict chili variety based on input"""
    try:
        # Validate input values against encoder classes
        for col, value in [('umur', umur), ('tinggi', tinggi), 
                          ('jumlah_daun', jumlah_daun), ('kondisi_daun', kondisi_daun)]:
            if value not in encoders[col].classes_:
                raise ValueError(f"Invalid value '{value}' for {col}. Acceptable values: {list(encoders[col].classes_)}")
        
        # Encode input features
        input_encoded = [
            encoders['umur'].transform([umur])[0],
            encoders['tinggi'].transform([tinggi])[0],
            encoders['jumlah_daun'].transform([jumlah_daun])[0],
            encoders['kondisi_daun'].transform([kondisi_daun])[0]
        ]
        
        # Make prediction
        input_df = pd.DataFrame([input_encoded], columns=['umur', 'tinggi', 'jumlah_daun', 'kondisi_daun'])
        proba = model.predict_proba(input_df)[0]
        kelas = label_encoder.inverse_transform([proba.argmax()])[0]
        
        return {
            'Rekomendasi': kelas,
            'Probabilitas': f"{proba.max()*100:.1f}%",
            'Detail': {k: f"{v*100:.1f}%" for k, v in zip(label_encoder.classes_, proba)}
        }
    except Exception as e:
        return {"Error": str(e)}

# ========================
# 4. Main Program
# ========================
def main():
    model, encoders, label_encoder = load_models()
    if None in [model, encoders, label_encoder]:
        return
    
    print("\n" + "="*40)
    print("SISTEM REKOMENDASI BIBIT CABAI UNGGUL")
    print("="*40)
    
    # Get validated user input
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
    else:
        print(f"Bibit terbaik: {hasil['Rekomendasi']} ({hasil['Probabilitas']})")
        print("\nProbabilitas Lengkap:")
        for jenis, prob in sorted(hasil['Detail'].items(), 
                                key=lambda x: float(x[1].replace('%', '')), 
                                reverse=True):
            print(f"- {jenis}: {prob}")

if __name__ == "__main__":
    main()