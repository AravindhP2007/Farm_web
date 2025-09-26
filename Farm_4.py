import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
from pymongo import MongoClient
import os
from deep_translator import GoogleTranslator
import joblib
import numpy as np
import pandas as pd

# ---------------- Firebase Init ----------------
if not firebase_admin._apps:
    try:
        cred_path = os.getenv("FIREBASE_CRED_JSON", "serviceAccountKey.json")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.sidebar.error("⚠ Firebase init failed. Running in demo mode (no real Auth).")

# ---------------- MongoDB Init ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["biosecure_portal"]

# ---------------- Translator (with cache) ----------------
LANGUAGES = {"English": "en", "தமிழ்": "ta", "हिंदी": "hi"}

if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None
if "translations" not in st.session_state:
    st.session_state.translations = {}
if "current_farmer" not in st.session_state:
    st.session_state.current_farmer = None  # store farmer temporarily for disease queries

def t(text: str):
    lang = st.session_state.lang
    if lang == "en":
        return text

    key = (text, lang)
    if key in st.session_state.translations:
        return st.session_state.translations[key]

    try:
        translated = GoogleTranslator(source="auto", target=lang).translate(text)
        st.session_state.translations[key] = translated
        return translated
    except:
        return text

# ---------------- ML Model Prediction Function ----------------
def predict_new(sample_dict):
    sample_df = pd.DataFrame([sample_dict])
    sample_df = pd.get_dummies(sample_df)

    feature_columns = joblib.load("feature_columns.pkl")
    sample_df = sample_df.reindex(columns=feature_columns, fill_value=0)

    model = joblib.load("multi_output_randomforest.pkl")
    encoders = joblib.load("label_encoders.pkl")

    pred = model.predict(sample_df)[0]

    decoded_pred = {
        col: encoders[col].inverse_transform([pred[i]])[0]
        for i, col in enumerate(encoders.keys())
    }
    return decoded_pred

# ---------------- Home Page ----------------
st.set_page_config(page_title="Digital Farm Management Portal", layout="wide")

st.markdown(
    f"<h3 style='text-align: center;'>{t('Digital Farm Management Portal')}</h3>",
    unsafe_allow_html=True,
)

# ---------------- Language Selector ----------------
lang_choice = st.sidebar.selectbox(
    "🌐 Select Language",
    list(LANGUAGES.keys()),
    index=list(LANGUAGES.values()).index(st.session_state.lang)
)
st.session_state.lang = LANGUAGES[lang_choice]

# ---------------- Auth Section ----------------
if st.session_state.user is None:
    menu = st.sidebar.radio("➡ Menu", [t("Home"), t("Login/Signup")])

    if menu == t("Home"):
        st.write(t("Welcome to the Biosecurity Portal for Pig & Poultry Farms."))

    elif menu == t("Login/Signup"):
        role = st.selectbox(t("Select Role"), ["Vet Shop", "Vet Doctor"])
        action = st.radio(t("Choose Action"), [t("Login"), t("Signup")])

        if action == t("Signup"):
            if role == "Vet Shop":
                shop_name = st.text_input(t("Shop Name"))
                owner_name = st.text_input(t("Owner Name"))
                phone = st.text_input(t("Phone Number"))
                address = st.text_area(t("Address"))
                location = st.text_input(t("Location"))

                if st.button(t("Register")):
                    if len(phone) != 10 or not phone.isdigit():
                        st.error(t("❌ Invalid number. Please enter a 10-digit phone number."))
                    else:
                        try:
                            if firebase_admin._apps:
                                auth.create_user(uid=phone, phone_number=f"+91{phone}")
                        except:
                            st.warning(t("Skipping Firebase user creation (demo mode)."))

                        db.vet_shops.insert_one({
                            "shop_name": shop_name,
                            "owner_name": owner_name,
                            "phone": phone,
                            "address": address,
                            "location": location
                        })
                        st.success(t("Vet Shop registered successfully!"))
                        st.session_state.user = {"shop_name": shop_name, "owner_name": owner_name,
                                                 "phone": phone, "address": address, "location": location}
                        st.session_state.role = "Vet Shop"

            elif role == "Vet Doctor":
                hospital_name = st.text_input(t("Hospital Name"))
                doctor_name = st.text_input(t("Doctor Name"))
                phone = st.text_input(t("Phone Number"))
                address = st.text_area(t("Address"))
                location = st.text_input(t("Location"))

                if st.button(t("Register")):
                    if len(phone) != 10 or not phone.isdigit():
                        st.error(t("❌ Invalid number. Please enter a 10-digit phone number."))
                    else:
                        try:
                            if firebase_admin._apps:
                                auth.create_user(uid=phone, phone_number=f"+91{phone}")
                        except:
                            st.warning(t("Skipping Firebase user creation (demo mode)."))

                        db.vet_doctors.insert_one({
                            "hospital_name": hospital_name,
                            "doctor_name": doctor_name,
                            "phone": phone,
                            "address": address,
                            "location": location
                        })
                        st.success(t("Vet Doctor registered successfully!"))
                        st.session_state.user = {"hospital_name": hospital_name, "doctor_name": doctor_name,
                                                 "phone": phone, "address": address, "location": location}
                        st.session_state.role = "Vet Doctor"

        elif action == t("Login"):
            phone = st.text_input(t("Phone Number"))
            if st.button(t("Login")):
                if len(phone) != 10 or not phone.isdigit():
                    st.error(t("❌ Invalid number. Please enter a 10-digit phone number."))
                else:
                    if role == "Vet Shop":
                        user = db.vet_shops.find_one({"phone": phone})
                    else:
                        user = db.vet_doctors.find_one({"phone": phone})

                    if user:
                        st.success(t("Login successful!"))
                        st.session_state.user = user
                        st.session_state.role = role
                    else:
                        st.error(t("User not found. Please Signup first."))

else:
    # ---------------- Dashboard ----------------
    col1, col2 = st.sidebar.columns([2, 1])
    with col2:
        if st.button(t("Logout")):
            st.session_state.user = None
            st.session_state.role = None

    with col1:
        if st.button(t("👤 Profile")):
            st.sidebar.subheader(t("Profile Details"))
            for key, value in st.session_state.user.items():
                st.sidebar.write(f"{t(key.capitalize())}: {value}")

    st.sidebar.success(f"{t('Logged in as')} {t(st.session_state.role)}")

    # ---------------- Vet Shop Dashboard ----------------
    if st.session_state.role == "Vet Shop":
        st.subheader(t("Vet Shop Dashboard"))

        # ✅ Farmer Registration Sidebar
        st.sidebar.subheader("👨‍🌾 Farmer Registration")
        farmer_name = st.sidebar.text_input("Farmer Name")
        farmer_phone = st.sidebar.text_input("Farmer Phone Number")

        if st.sidebar.button("➕ Add Farmer"):
            if len(farmer_phone) != 10 or not farmer_phone.isdigit():
                st.sidebar.error("❌ Invalid phone number.")
            else:
                db.farmers.insert_one({
                    "shop_name": st.session_state.user["shop_name"],
                    "farmer_name": farmer_name,
                    "farmer_phone": farmer_phone
                })
                st.session_state.current_farmer = {"farmer_name": farmer_name, "farmer_phone": farmer_phone}
                st.sidebar.success("✅ Farmer added successfully!")

        st.write(t("🐖 Disease Prediction Tool"))
        species = st.selectbox(t("Select Species"), ["Pig", "Poultry"])
        symptoms = st.text_area(t("Enter Symptoms"))
        days_not_well = st.number_input(t("Number of Days Not Well"), min_value=1, max_value=60, step=1)

        if st.button(t("🔍 Predict Disease")):
            try:
                sample = {
                    "species": species,
                    "clinical_signs": symptoms,
                    "days_not_well": days_not_well
                }
                prediction = predict_new(sample)
                st.success(f"✅ {t('Predicted Disease')}: {prediction}")

                # ✅ Save query with farmer details
                query_data = {
                    "shop_name": st.session_state.user["shop_name"],
                    "phone": st.session_state.user["phone"],
                    "location": st.session_state.user["location"],
                    "input_data": sample,
                    "prediction": prediction,
                    "timestamp": pd.Timestamp.now()
                }
                if st.session_state.current_farmer:
                    query_data["farmer"] = st.session_state.current_farmer

                db.disease_queries.insert_one(query_data)

            except Exception as e:
                st.error(f"{t('Model Error')}: {str(e)}")

        st.write(t("📍 Browse Vet Doctors by District"))
        districts = ["Erode", "Salem", "Namakkal", "Coimbatore", "Madurai", "Tirupur"]
        district = st.selectbox(t("Select District"), districts)
        if st.button(t("Show Doctors")):
            doctors = db.vet_doctors.find({"location": district})
            for doc in doctors:
                st.write(f"👨‍⚕️ {doc['doctor_name']} - {doc['phone']} ({doc['location']})")

    # ---------------- Vet Doctor Dashboard ----------------
    elif st.session_state.role == "Vet Doctor":
        st.subheader(t("Vet Doctor Dashboard"))

        st.write(t("🐖 Disease Prediction Tool"))
        species = st.selectbox(t("Select Species"), ["Pig", "Poultry"])
        symptoms = st.text_area(t("Enter Symptoms"))
        days_not_well = st.number_input(t("Number of Days Not Well"), min_value=1, max_value=60, step=1)

        if st.button(t("🔍 Predict Disease")):
            try:
                sample = {
                    "species": species,
                    "clinical_signs": symptoms,
                    "days_not_well": days_not_well
                }
                prediction = predict_new(sample)
                st.success(f"✅ {t('Predicted Disease')}: {prediction}")
            except Exception as e:
                st.error(f"{t('Model Error')}: {str(e)}")

        st.write("📊 Disease Queries from Vet Shops in Your District")
        location = st.session_state.user.get("location")
        queries = db.disease_queries.find({"location": location}).sort("timestamp", -1)

        for q in queries:
            farmer_info = ""
            if "farmer" in q:
                farmer_info = f"👨‍🌾 Farmer: {q['farmer']['farmer_name']} ({q['farmer']['farmer_phone']})\n"
            st.info(f"""
            🏪 Shop: {q['shop_name']} ({q['phone']})  
            📍 Location: {q['location']}  
            {farmer_info}❓ Symptoms: {q['input_data']['clinical_signs']}  
            🐖 Species: {q['input_data']['species']}  
            ⏳ Days Not Well: {q['input_data']['days_not_well']}  
            🔮 Prediction: {q['prediction']}  
            🕒 Time: {q['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}  
            """)

        st.write(t("📍 Browse Vet Shops by District"))
        districts = ["Erode", "Salem", "Namakkal", "Coimbatore", "Madurai", "Tirupur"]
        district = st.selectbox(t("Select District"), districts)
        if st.button(t("Show Shops")):
            shops = db.vet_shops.find({"location": district})
            for shop in shops:
                st.write(f"🏪 {shop['shop_name']} - {shop['phone']} ({shop['location']})")
