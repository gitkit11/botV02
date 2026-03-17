import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_key(name, url, headers=None):
    key = os.getenv(name)
    if not key:
        print(f"❌ {name}: Не найден в .env")
        return
    try:
        if "{key}" in url:
            final_url = url.replace("{key}", key)
            r = requests.get(final_url, timeout=10)
        else:
            final_headers = headers or {}
            if "Authorization" in final_headers:
                final_headers["Authorization"] = final_headers["Authorization"].replace("{key}", key)
            if "x-apisports-key" in final_headers:
                final_headers["x-apisports-key"] = final_headers["x-apisports-key"].replace("{key}", key)
            r = requests.get(url, headers=final_headers, timeout=10)
        
        if r.status_code == 200:
            print(f"✅ {name}: OK")
        else:
            print(f"❌ {name}: Ошибка {r.status_code} ({r.text[:50]}...)")
    except Exception as e:
        print(f"❌ {name}: Ошибка {e}")

if __name__ == "__main__":
    print("--- Упрощенная проверка API ---")
    test_key("THE_ODDS_API_KEY", "https://api.the-odds-api.com/v4/sports/?apiKey={key}")
    test_key("THE_ODDS_API_KEY", "https://api.the-odds-api.com/v4/sports/?apiKey={key}")
    test_key("API_FOOTBALL_KEY", "https://v3.football.api-sports.io/status", headers={"x-apisports-key": "{key}"})
    test_key("GROQ_API_KEY", "https://api.groq.com/openai/v1/models", headers={"Authorization": "Bearer {key}"})
    test_key("PANDASCORE_API_KEY", "https://api.pandascore.co/csgo/teams?per_page=1", headers={"Authorization": "Bearer {key}"})
