import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# ตั้งค่าหน้าเว็บ
# -----------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 ทำนายการรอดชีวิตบนเรือไททานิค")
st.write("กรอกข้อมูลผู้โดยสารด้านล่าง แล้วกดปุ่มเพื่อให้โมเดลทำนายผล")

# -----------------------------
# โหลดโมเดล (ต้องวางไฟล์ .joblib ไว้โฟลเดอร์เดียวกับ app.py)
# -----------------------------
MODEL_PATH = "titanic_tree (3).joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"ไม่พบไฟล์โมเดล '{MODEL_PATH}' กรุณาตรวจสอบว่าไฟล์นี้อยู่ในโฟลเดอร์เดียวกับ app.py"
    )
    st.stop()

# -----------------------------
# ฟอร์มรับข้อมูลจากผู้ใช้
# โมเดลนี้ถูกฝึกด้วย feature: Pclass, Sex_female, Age, Fare, Family
# -----------------------------
st.subheader("ข้อมูลผู้โดยสาร")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("ชั้นโดยสาร (Pclass)", options=[1, 2, 3], index=2)
    sex = st.radio("เพศ", options=["หญิง", "ชาย"])
    age = st.number_input("อายุ (Age)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

with col2:
    fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, value=32.0, step=1.0)
    family = st.number_input(
        "จำนวนสมาชิกครอบครัวที่มาด้วย (Family = พี่น้อง/คู่สมรส + พ่อแม่/ลูก)",
        min_value=0, max_value=15, value=0, step=1,
    )

sex_female = 1 if sex == "หญิง" else 0

# -----------------------------
# ทำนายผล
# -----------------------------
if st.button("ทำนายผล"):
    # ต้องเรียงคอลัมน์ให้ตรงกับตอนฝึกโมเดล: Pclass, Sex_female, Age, Fare, Family
    input_df = pd.DataFrame(
        [[pclass, sex_female, age, fare, family]],
        columns=["Pclass", "Sex_female", "Age", "Fare", "Family"],
    )

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("ผลการทำนาย")
    if prediction == 1:
        st.success(f"✅ รอดชีวิต (ความน่าจะเป็น {proba[1]*100:.1f}%)")
    else:
        st.error(f"❌ ไม่รอดชีวิต (ความน่าจะเป็นไม่รอด {proba[0]*100:.1f}%)")

    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(input_df)
