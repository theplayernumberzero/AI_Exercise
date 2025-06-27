import gradio as gr
import os
from dotenv import load_dotenv
from typing import List, Tuple

# Gemini API için gerekli kütüphane (örnek: google-generativeai)
import google.generativeai as genai

# .env dosyasını yükle
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)

# Model nesnesi oluştur
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
response = model.generate_content("Chatbot hazır mı?")
print(response.text)

# --- Özelleştirilebilir Sistem Promptu ---
SYSTEM_PROMPT = """
Serseri bir iş arkadaşı gibi davran.
"""

# --- Hafıza için geçmişi tutan fonksiyon ---
def build_history(messages):
    history = SYSTEM_PROMPT + "\n"
    for msg in messages:
        if msg["role"] == "user":
            history += f"Kullanıcı: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history += f"Asistan: {msg['content']}\n"
    return history

# --- Chat fonksiyonu ---
def chatbot_fn(user_input, history):
    if any(phrase in user_input.lower() for phrase in ["önceki soruyu", "son soruyu", "bir önceki soruyu"]):
        last_user_msg = None
        for msg in reversed(history):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        if last_user_msg:
            prompt = build_history(history[:-2]) + f"Kullanıcı: {last_user_msg}\nAsistan: Lütfen bu mesajı daha detaylı açıkla."
        else:
            prompt = SYSTEM_PROMPT + "\nKullanıcı: (Önceki soru yok)\nAsistan: Açıklayacak bir önceki soru bulunamadı."
    else:
        prompt = build_history(history) + f"Kullanıcı: {user_input}\nAsistan:"

    response = model.generate_content(prompt)
    bot_reply = response.text.strip()
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_reply})
    return history, history

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# Özelleştirilebilir Gemini Chatbot")
    gr.Markdown("Sistem promptunu ve davranışını koddan değiştirebilirsin.")
    chatbot = gr.Chatbot(type='messages')
    state = gr.State([])  # Hafıza için
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Mesajınızı yazın...")
    
    def respond(message, history):
        history, _ = chatbot_fn(message, history)
        return history, history

    txt.submit(respond, [txt, state], [chatbot, state])

if __name__ == "__main__":
    demo.launch()
