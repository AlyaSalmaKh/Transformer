import streamlit as st
import joblib
import json
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import numpy as np

# === Download Files from Hugging Face ===
REPO_ID = "Alya83/Transformer"

@st.cache_resource
def load_files_from_hf():
    """Download semua file yang diperlukan dari Hugging Face"""
    try:
        st.info("📥 Mendownload file dari Hugging Face...")
        
        # Download config.json
        config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        # Download label2id.json
        label2id_path = hf_hub_download(repo_id=REPO_ID, filename="label2id.json")
        with open(label2id_path) as f:
            label2id = json.load(f)
        
        # Download logreg.pkl
        model_path = hf_hub_download(repo_id=REPO_ID, filename="logreg.pkl")
        clf = joblib.load(model_path)
        
        return config, label2id, clf
        
    except Exception as e:
        st.error(f"❌ Gagal memuat file dari Hugging Face: {e}")
        st.stop()

# === Load backbone model & tokenizer ===
@st.cache_resource
def load_transformer_model(config):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(config["transformer_model"])
        model = AutoModel.from_pretrained(config["transformer_model"]).to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ Gagal memuat model transformer: {e}")
        st.stop()

# === Helper functions ===
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.no_grad()
def embed_text(texts, max_len=160):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    ).to(device)

    out = model(**enc)
    pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])
    return pooled.cpu().numpy()

def predict_text(text: str):
    emb = embed_text([text])
    pred = clf.predict(emb)[0]
    label = pred if isinstance(pred, str) else id2label[pred]
    return label

def predict_with_confidence(text: str):
    emb = embed_text([text])
    pred = clf.predict(emb)[0]
    probabilities = clf.predict_proba(emb)[0]
    
    label = pred if isinstance(pred, str) else id2label[pred]
    confidence = max(probabilities)
    
    # Get all confidence scores
    confidence_scores = {}
    for i, prob in enumerate(probabilities):
        label_name = id2label[i] if i in id2label else str(i)
        confidence_scores[label_name] = prob
    
    return label, confidence, confidence_scores

# === Streamlit UI ===
st.set_page_config(page_title="Emotion Classifier", page_icon="😊", layout="centered")

st.title("🎭 Emotion Classifier")
st.write("Masukkan teks untuk diklasifikasikan ke dalam emosi.")

# Load files dengan progress indicator
with st.spinner("Memuat model dari Hugging Face..."):
    config, label2id, clf = load_files_from_hf()
    id2label = {v: k for k, v in label2id.items()}
    tokenizer, model, device = load_transformer_model(config)

st.success("✅ Semua model berhasil dimuat!")

# Tampilkan info model
with st.expander("ℹ️ Info Model"):
    st.write(f"**Transformer Model:** {config['transformer_model']}")
    st.write(f"**Device:** {device}")
    st.write(f"**Labels:** {list(label2id.keys())}")
    st.write(f"**Source:** [Hugging Face - Alya83/Transformer](https://huggingface.co/Alya83/Transformer)")

# Input teks
user_input = st.text_area("Teks input:", "", height=100, placeholder="Masukkan teks di sini...")

if st.button("Prediksi", type="primary"):
    if user_input.strip():
        with st.spinner("Sedang menganalisis emosi..."):
            try:
                # Prediksi dengan confidence
                result, confidence, confidence_scores = predict_with_confidence(user_input)
                
                # Tampilkan hasil utama
                st.success(f"**Prediksi Emosi:** {result}")
                st.info(f"**Tingkat Keyakinan:** {confidence:.2%}")
                
                # Tampilkan progress bar confidence
                st.progress(float(confidence))
                
                # Tampilkan confetti jika confidence tinggi
                if confidence > 0.7:  # Jika confidence > 70%
                    st.balloons()
                    st.success("🎉 Prediksi sangat confident!")
                elif confidence > 0.5:  # Jika confidence > 50%
                    st.snow()
                    st.info("❄️ Prediksi cukup confident!")
                
                # Tampilkan semua confidence scores
                st.write("**Detail Tingkat Keyakinan:**")
                for label, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                    col1, col2 = st.columns([3, 7])
                    with col1:
                        st.write(f"**{label}:**")
                    with col2:
                        st.progress(float(score), text=f"{score:.2%}")
                    
            except Exception as e:
                st.error(f"Error dalam prediksi: {e}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")

# Footer
st.markdown("---")
st.caption("Dibuat dengan Streamlit & Transformers | Model oleh Alya83")