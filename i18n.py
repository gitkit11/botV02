# -*- coding: utf-8 -*-
"""
i18n.py — Переводы для онбординга и главного меню.
Поддерживаемые языки: ru, en
"""

STRINGS = {
    "ru": {
        # Онбординг
        "lang_select": (
            "🌐 <b>Выбери язык</b>\n\n"
            "Choose your language:"
        ),
        "btn_ru": "🇷🇺 Русский",
        "btn_en": "🇬🇧 English",

        "appear_1": "🐉",
        "appear_2": "<i>Где-то между числами и инстинктом\nрождается единственно верное решение...</i>",
        "appear_3": "🔥 <b>CHIMERA AI</b> 🔥\n<i>Три разума. Один сигнал.</i>",

        "legend": (
            "🐍🦁🐐 <b>CHIMERA AI</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>Три головы. Три измерения реальности.</b>\n\n"
            "🦁 <b>Лев — МАТЕМАТИКА</b>\n"
            "   ELO · Пуассон · Dixon-Coles · нейросеть\n"
            "   100 000+ матчей. Вероятности, шансы, перевес —\n"
            "   всё просчитано до последней цифры.\n\n"
            "🐐 <b>Козёл — КОНТЕКСТ</b>\n"
            "   Новости · травмы · движение линий · рынок.\n"
            "   То, что не видно в цифрах — живёт здесь.\n"
            "   Форма, мотивация, инсайд — всё учтено.\n\n"
            "🐍 <b>Змея — ЛОГИКА</b>\n"
            "   Три независимых ИИ-агента разбирают каждый матч.\n"
            "   Связывают числа с реальностью, взвешивают риски,\n"
            "   находят противоречия. Не сошлись — сигнал не выходит.\n\n"
            "🧬 <b>CHIMERA Score — финальный вердикт</b>\n"
            "   Сложный расчёт: каждому фактору — свой вес и балл.\n"
            "   ELO, форма, ценность кэфа, движение линий...\n"
            "   Всё это фильтруют 3 ИИ-агента. Только лучшее — выходит.\n\n"
            "<i>Букмекеры зарабатывают на тех, кто ставит на эмоциях.\n"
            "Chimera балансирует между числами, фактами и разумом.</i>"
        ),

        "features": (
            "⚡ <b>Что умеет CHIMERA:</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🧮 <b>Математическая модель без аналогов</b>\n"
            "   Пуассон + ELO + Dixon-Coles + нейросеть\n"
            "   Обучена на <b>100 000+</b> реальных матчей —\n"
            "   футбол, баскетбол, теннис, хоккей, CS2\n\n"
            "🤖 <b>Три уровня ИИ — не просто бот</b>\n"
            "   · Химера (ИИ-агент) — математический аналитик\n"
            "   · Тень (ИИ-агент) — независимый контраргумент\n"
            "   · MetaLearner (ИИ-агент) — самообучается на реальных результатах\n"
            "   Не сошлись — сигнал не выходит. Точка.\n\n"
            "🎯 <b>Только ценные ставки (EV+)</b>\n"
            "   Никаких прогнозов ради прогнозов\n"
            "   Только там где математика говорит «да»\n\n"
            "🏆 <b>5 видов спорта — один интерфейс</b>\n"
            "   Футбол · Баскетбол · Теннис · Хоккей · CS2\n"
            "   Аналогов такого охвата в Telegram нет\n\n"
            "🐉 <b>Личный советник Химера</b>\n"
            "   Задай вопрос — получи прямой ответ\n"
            "   Без воды, без рекламы, без лжи"
        ),

        "benefits": (
            "🔑 <b>Почему CHIMERA — это другой уровень:</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✅ <b>Не прогнозы — решения с цифрами</b>\n"
            "   Вероятность, EV, Kelly — всё открыто.\n"
            "   Ты видишь почему, а не просто «ставь».\n\n"
            "✅ <b>100 000+ матчей в основе</b>\n"
            "   Модель обучена на реальных данных всех лиг.\n"
            "   Не на тестовых — на боевых.\n\n"
            "✅ <b>Честная статистика в реальном времени</b>\n"
            "   Каждый прогноз записывается автоматически.\n"
            "   Никаких задним числом исправленных пиков.\n\n"
            "✅ <b>Контрарные сигналы</b>\n"
            "   Находим ценность там где букмекер ошибся.\n"
            "   Даже на андердогов с высоким EV.\n\n"
            "🎁 <b>Сейчас — бесплатный доступ</b>\n"
            "   Пользуйся пока открыто."
        ),

        "enter_btn": "🐉 Войти в CHIMERA",
        "enter_msg": (
            "🐍🦁🐐 <b>CHIMERA AI — готова к работе</b>\n\n"
            "Выбери спорт или посмотри сигналы дня.\n"
            "<i>Удачи — хотя с нами это уже не удача.</i>"
        ),

        # Главное меню
        "btn_signals":    "📡 Сигналы дня",
        "btn_express":    "🎯 Экспресс (бета)",
        "btn_football":   "⚽ Футбол",
        "btn_tennis":     "🎾 Теннис",
        "btn_cs2":        "🎮 CS2 (бета)",
        "btn_basketball": "🏀 Баскетбол",
        "btn_hockey":     "🏒 Хоккей",
        "btn_stats":      "📊 Статистика",
        "btn_cabinet":    "👤 Кабинет",
        "btn_vip":        "💎 Подписка Химера",
        "btn_support":    "💬 Поддержка",
        "btn_hunt":       "🔥 Охота Химеры 📈",
    },

    "en": {
        # Onboarding
        "lang_select": (
            "🌐 <b>Choose language</b>\n\n"
            "Выбери язык:"
        ),
        "btn_ru": "🇷🇺 Русский",
        "btn_en": "🇬🇧 English",

        "appear_1": "🐉",
        "appear_2": "<i>Somewhere between numbers and instinct\nlies the only right decision...</i>",
        "appear_3": "🔥 <b>CHIMERA AI</b> 🔥\n<i>Three minds. One signal.</i>",

        "legend": (
            "🐍🦁🐐 <b>CHIMERA AI</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>Three heads. Three dimensions of reality.</b>\n\n"
            "🦁 <b>Lion — MATHEMATICS</b>\n"
            "   ELO · Poisson · Dixon-Coles · neural network\n"
            "   100,000+ matches. Probabilities, edges, value —\n"
            "   every number calculated to the last decimal.\n\n"
            "🐐 <b>Goat — CONTEXT</b>\n"
            "   News · injuries · line movement · market.\n"
            "   What numbers can't see — lives here.\n"
            "   Form, motivation, insider signals — all factored in.\n\n"
            "🐍 <b>Snake — LOGIC</b>\n"
            "   Three independent AI agents dissect each match.\n"
            "   They link numbers to reality, weigh risks,\n"
            "   find contradictions. Disagree — no signal fires.\n\n"
            "🧬 <b>CHIMERA Score — the final verdict</b>\n"
            "   A complex calculation: every factor has its own weight.\n"
            "   ELO, form, value, line movement...\n"
            "   All of it filtered by 3 AI agents. Only the best fires.\n\n"
            "<i>Bookmakers profit from those who bet on emotions.\n"
            "Chimera balances numbers, facts and reason.</i>"
        ),

        "features": (
            "⚡ <b>What CHIMERA does:</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🧮 <b>A mathematical model with no equal</b>\n"
            "   Poisson + ELO + Dixon-Coles + neural network\n"
            "   Trained on <b>100,000+</b> real matches —\n"
            "   football, basketball, tennis, hockey, CS2\n\n"
            "🤖 <b>Three AI layers — not just a bot</b>\n"
            "   · Chimera (AI agent) — mathematical analyst\n"
            "   · Shadow (AI agent) — independent counterargument\n"
            "   · MetaLearner (AI agent) — self-trains on real outcomes\n"
            "   Disagree — no signal fires. Period.\n\n"
            "🎯 <b>Only value bets (EV+)</b>\n"
            "   No predictions just for the sake of it\n"
            "   Only where the math says yes\n\n"
            "🏆 <b>5 sports — one interface</b>\n"
            "   Football · Basketball · Tennis · Hockey · CS2\n"
            "   Nothing like this exists in Telegram\n\n"
            "🐉 <b>Personal advisor Chimera</b>\n"
            "   Ask — get a direct answer\n"
            "   No fluff, no ads, no lies"
        ),

        "benefits": (
            "🔑 <b>Why CHIMERA is a different level:</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✅ <b>Not predictions — decisions with numbers</b>\n"
            "   Probability, EV, Kelly — all visible.\n"
            "   You see the why, not just «bet this».\n\n"
            "✅ <b>100,000+ matches behind every signal</b>\n"
            "   Model trained on real data across all leagues.\n"
            "   Not test data — live battle data.\n\n"
            "✅ <b>Honest stats, updated live</b>\n"
            "   Every prediction logged automatically.\n"
            "   No backdated correct picks. Ever.\n\n"
            "✅ <b>Contrarian signals</b>\n"
            "   We find value where the bookmaker slipped.\n"
            "   Even underdogs with high EV.\n\n"
            "🎁 <b>Free access — right now</b>\n"
            "   Use it while it's open."
        ),

        "enter_btn": "🐉 Enter CHIMERA",
        "enter_msg": (
            "🐍🦁🐐 <b>CHIMERA AI — ready</b>\n\n"
            "Choose a sport or check today's signals.\n"
            "<i>Good luck — though with us it's no longer luck.</i>"
        ),

        # Main menu
        "btn_signals":    "📡 Daily Signals",
        "btn_express":    "🎯 Express (beta)",
        "btn_football":   "⚽ Football",
        "btn_tennis":     "🎾 Tennis",
        "btn_cs2":        "🎮 CS2 (beta)",
        "btn_basketball": "🏀 Basketball",
        "btn_hockey":     "🏒 Hockey",
        "btn_stats":      "📊 Statistics",
        "btn_cabinet":    "👤 Profile",
        "btn_vip":        "💎 Chimera Subscription",
        "btn_support":    "💬 Support",
        "btn_hunt":       "🔥 Chimera Hunt 📈",
    },
}


def t(key: str, lang: str = "ru") -> str:
    """Возвращает перевод строки по ключу и языку."""
    return STRINGS.get(lang, STRINGS["ru"]).get(key, STRINGS["ru"].get(key, key))
