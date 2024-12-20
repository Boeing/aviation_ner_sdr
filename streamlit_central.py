'''
    Boeing Proprietary.
    Developed by Daniel Whyatt, Boeing AI & ML
'''

import streamlit as st
from run_newnerzero import NERTagging

# to run from command line: streamlit run streamlit_central
# tasklist | findstr streamlit
# taskkill /PID 26564 /F

# TODO: refactor to make the input path as optional,  if model path is not passed in use Huggingface
@st.cache_resource
def load_model():
    model_path = r"\NERStreamlit\models\nunerzero_bio"
    tagger = NERTagging(model_path)
    return tagger

tagger = load_model()

label_colors = {
    "prod": "#CB4335",  # Dark Salmon
    "loc": "#33cc33",   # Green
    "pcon": "#33ccff",  # Light Blue
    "sit": "#FFD700",   # Gold
    "act": "#FF69B4",   # Hot Pink
    "bird": "#FF5733",  # Orange Red
    "flt": "#9370DB"    # Medium Purple
}

# Function to create HTML-styled text with colors based on labels
def style_text(tokens_with_labels):

    styled_sentence = ""

    for token, label in tokens_with_labels:
        if label in label_colors:
            color = label_colors[label]
            styled_sentence += f"<span style='color: {color}; font-weight: bold; font-size:20px;'>{token}</span> "

        else:
            styled_sentence += f"{token} "

    return styled_sentence.strip()


text = st.text_input("Enter a sentence:")
if text:
    labeled_text = tagger.ner_label_main(text, strip_bi=True)
    styled_text = style_text(labeled_text)
    st.markdown(styled_text, unsafe_allow_html=True)
    st.session_state.text = ""

st.markdown("<div style='padding-top: 60px;'></div>", unsafe_allow_html=True)
# st.markdown("<h3 style='font-size:16px;'>Entity Label Colors</h3>", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; font-weight: bold; font-size:20px;'>Entity Color Legend</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

for label, color in label_colors.items():
    st.markdown(
        f"<div style='text-align: center;'>"
        f"<span style='color:{color}; font-weight:bold; font-size:16px;'>{label}:</span> "
        f"<span style='background-color:{color}; color:white; padding:2px 8px; border-radius:3px;'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True
    )