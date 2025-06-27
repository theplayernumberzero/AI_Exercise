# 'openai' kütüphanesinin kurulu olması gerekir. Kurulu değilse: pip install openai
try:
    import openai
except ImportError:
    openai = None
from typing import List, Dict, Callable, Tuple
import requests

# LLM API anahtarlarını buraya ekleyebilirsiniz
OPENAI_API_KEY = None  # "sk-..."
ANTHROPIC_API_KEY = None  # "claude-..."
GOOGLE_API_KEY = None  # "..."
DEEPSEEK_API_KEY = None  # "..."

# LLM çağrısı yapan fonksiyonlar (örnek olarak OpenAI, diğerleri eklenebilir)
def ask_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    if not OPENAI_API_KEY or openai is None:
        return "[OpenAI cevabı alınamadı]"
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()

# Diğer LLM fonksiyonlarını da benzer şekilde ekleyebilirsiniz
def ask_dummy(prompt: str, model: str = "dummy") -> str:
    # Burada başka bir LLM API'si ile entegre edebilirsiniz
    return f"[Dummy cevap: {prompt[:20]}...]"

def ask_openai_gpt4o(prompt):
    if not OPENAI_API_KEY:
        return "[OpenAI API anahtarı eksik]"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    return f"[OpenAI Hata: {resp.text}]"

def ask_anthropic_sonnet(prompt):
    if not ANTHROPIC_API_KEY:
        return "[Anthropic API anahtarı eksik]"
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        # Claude API'si dökümantasyonuna göre cevap formatı değişebilir
        try:
            return resp.json()["content"][0]["text"].strip()
        except Exception:
            return str(resp.json())
    return f"[Anthropic Hata: {resp.text}]"

def ask_google_gemini(prompt):
    if not GOOGLE_API_KEY:
        return "[Google Gemini API anahtarı eksik]"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            return str(resp.json())
    return f"[Gemini Hata: {resp.text}]"

def ask_deepseek(prompt):
    if not DEEPSEEK_API_KEY:
        return "[DeepSeek API anahtarı eksik]"
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return str(resp.json())
    return f"[DeepSeek Hata: {resp.text}]"

def ask_judge(prompt):
    # Hakem olarak GPT-4o kullanılıyor, istersen başka bir modelle değiştirebilirsin
    return ask_openai_gpt4o(prompt)

def main():
    # 1. Konu belirle
    topic = "Yapay zekanın gelecekteki rolü nedir?"

    # 2. Hakem LLM ile soru oluştur
    question = ask_judge(f"'{topic}' hakkında LLM'lerin cevaplarını karşılaştırmak için anlamlı bir soru oluştur. Sadece soruyu döndür.")

    # 3. LLM'lere sor
    llms = {
        "gpt-4o-mini": ask_openai_gpt4o,
        "sonnet": ask_anthropic_sonnet,
        "gemini": ask_google_gemini,
        "deepseek": ask_deepseek
    }
    answers = []
    for name, llm_func in llms.items():
        answer = llm_func(question)
        answers.append((name, answer))

    # 4. Cevapları birleştir
    merged = "\n\n".join([f"Model: {name}\nCevap: {ans}" for name, ans in answers])
    merged_prompt = f"Aşağıda farklı LLM'lerin verdiği cevaplar var. En iyi cevabı seç ve nedenini açıkla.\n\n{merged}"

    # 5. Hakem LLM ile en iyi cevabı seçtir
    best = ask_judge(merged_prompt)

    # 6. Sonuçları yazdır
    print(f"Oluşturulan soru: {question}\n")
    print(f"Birleştirilmiş cevaplar:\n{merged}\n")
    print(f"En iyi cevap değerlendirmesi:\n{best}")

if __name__ == "__main__":
    main()
