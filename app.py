import json, pickle
import numpy as np, pandas as pd, streamlit as st
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Horizon", page_icon="◐", layout="wide")
FEATS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

@st.cache_resource
def load():
    pipe = pickle.load(open("model.pkl", "rb"))
    df = pd.read_csv("data/diabetes.csv")
    nn = NearestNeighbors(n_neighbors=30).fit(pipe[:-1].transform(df[FEATS]))
    return pipe, df, nn, json.load(open("metrics.json"))

pipe, df, nn, M = load()
risk = lambda r: float(pipe.predict_proba(pd.DataFrame([r])[FEATS])[0, 1])

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@300;400;500&display=swap');
#MainMenu, header, footer {visibility:hidden}
.stApp {background:radial-gradient(900px 480px at 18% -8%,#18301f 0%,#0a0d0b 62%);font-family:'Inter Tight',sans-serif}
.block-container {max-width:1120px;padding-top:2.2rem}
.tags span {display:inline-block;border:1px solid #ffffff26;border-radius:99px;padding:.3rem .9rem;margin:0 .45rem .45rem 0;font-size:.82rem;color:#c9d3cb}
.hero h1 {font-weight:300;font-size:clamp(2.8rem,7.5vw,6.2rem);line-height:.96;letter-spacing:-.035em;margin:1.4rem 0 1rem}
.hero h1 em, .big, .serif {font-family:'Instrument Serif',serif;font-style:italic;font-weight:400;color:#9ad1a5;letter-spacing:-.01em}
.hero p {max-width:34rem;color:#a9b5ac;font-size:1.08rem;line-height:1.55}
[data-testid="stVerticalBlockBorderWrapper"], .glass {border-radius:30px !important;border:1px solid #ffffff17 !important;background:linear-gradient(160deg,#ffffff0d,#ffffff03);backdrop-filter:blur(14px)}
.glass {padding:1.7rem 1.8rem}
.big {font-size:clamp(5rem,12vw,8.5rem);line-height:.9;margin:.2rem 0}
.sea {height:7px;border-radius:9px;margin:1.4rem 0 .5rem;position:relative;background:linear-gradient(90deg,#4f9a63,#c9c46a 55%,#e8a838)}
.sea i {position:absolute;top:-6px;width:19px;height:19px;border-radius:50%;background:#fff;box-shadow:0 0 22px #ffffffaa;transform:translateX(-50%)}
.row {display:flex;justify-content:space-between;gap:1rem;font-size:.95rem;color:#c9d3cb;margin-top:1rem}
.bar {height:8px;border-radius:9px;background:#ffffff14;margin-top:.45rem}
.bar b {display:block;height:8px;border-radius:9px;background:linear-gradient(90deg,#4f9a63,#9ad1a5)}
.fine {color:#7f8c83;font-size:.82rem;line-height:1.6;max-width:46rem}
</style>""", unsafe_allow_html=True)

st.markdown(f"""<div class="hero"><div class="tags"><span>Trained on {M['patients']} real patient records</span>
<span>Cross-validated AUC {M['auc']:.2f}</span><span>Open source</span></div>
<h1>See how far you are<br>from <em>the line</em>,<br>and what moves it.</h1>
<p>Most risk calculators give you a number and stop. Horizon shows where you sit among similar people, then which changes would shift your risk the most.</p></div>""", unsafe_allow_html=True)

st.write("")
left, right = st.columns([5, 6], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Your numbers")
        age = st.slider("Age", 21, 81, 35)
        glu = st.slider("Blood glucose (mg/dL, fasting if you can)", 50, 250, 110)
        bmi = st.slider("BMI", 15.0, 60.0, 27.0, 0.5)
        bp = st.slider("Diastolic blood pressure (mmHg)", 40, 130, 74)
        preg = st.slider("Pregnancies", 0, 15, 1)
        fam = st.select_slider("Family history of diabetes",
                               ["None known", "One relative", "Several relatives"])
        with st.expander("Optional lab values"):
            have = st.checkbox("I have insulin and skin-fold measurements")
            ins = st.number_input("Insulin (µU/mL)", 0, 900, 100, disabled=not have)
            skin = st.number_input("Skin fold (mm)", 0, 99, 20, disabled=not have)

row = dict(Pregnancies=preg, Glucose=glu, BloodPressure=bp,
           SkinThickness=skin if have else np.nan, Insulin=ins if have else np.nan,
           BMI=bmi, DiabetesPedigreeFunction={"None known": .25, "One relative": .5, "Several relatives": 1.0}[fam], Age=age)
p0 = risk(row)
peers = df.iloc[nn.kneighbors(pipe[:-1].transform(pd.DataFrame([row])[FEATS]))[1][0]]
share = peers.Outcome.mean()
band = "Lower" if p0 < .25 else "Moderate" if p0 < .5 else "Elevated" if p0 < .7 else "High"

moves = [("Glucose", -20, 70, "Glucose 20 mg/dL lower"), ("BMI", -3, 18.5, "BMI 3 points lower"),
         ("BloodPressure", -10, 60, "Blood pressure 10 mmHg lower")]
levers = []
for col, d, floor, label in moves:
    if row[col] + d >= floor:
        gain = p0 - risk({**row, col: row[col] + d})
        if gain > .005: levers.append((gain, label))
levers.sort(reverse=True)

lv = "".join(f"""<div class="row"><span>{l}</span><span>−{g*100:.0f} points</span></div>
<div class="bar"><b style="width:{min(100, g/levers[0][0]*100):.0f}%"></b></div>""" for g, l in levers) \
    or '<p class="fine">No single change in this model moves your risk by more than half a point.</p>'
nudge = ("Risk is not a diagnosis. At this level a fasting glucose or HbA1c test is worth asking a clinician about."
         if p0 >= .5 else "Risk is not a diagnosis. Keep tracking it, and retest if your numbers change.")

with right:
    st.markdown(f"""<div class="glass"><div class="serif" style="font-size:1.5rem">{band} risk</div>
<div class="big">{p0*100:.0f}<span style="font-size:.4em">%</span></div>
<div class="sea"><i style="left:{p0*100:.0f}%"></i></div>
<p style="color:#c9d3cb;margin-top:1.2rem">Among the 30 people in the dataset closest to your profile, <b>{share*100:.0f}%</b> had diabetes.</p>
<div class="serif" style="font-size:1.5rem;margin-top:1.8rem">What would move it</div>{lv}
<p class="fine" style="margin-top:1.4rem">{nudge}</p></div>""", unsafe_allow_html=True)

st.markdown(f"""<p class="fine" style="margin-top:2.5rem">Honest limits: the model learned from {M['patients']} women of Pima Indian heritage aged 21 and over,
so it may not fit everyone. Changes shown are statistical associations in that data, not proven effects. Cross-validated accuracy {M['accuracy']*100:.0f}%,
recall {M['recall']*100:.0f}%. Missing insulin or skin-fold values are filled with the dataset median. Built for learning, not medical advice.</p>""", unsafe_allow_html=True)
