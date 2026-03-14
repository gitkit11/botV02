import asyncio
import logging
import re
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

async def get_hltv_odds_async(team1: str, team2: str):
    """
    Парсит коэффициенты с HLTV.org для указанных команд.
    Поддерживает поиск по предстоящим матчам и прямым ссылкам.
    """
    logger.info(f"Searching HLTV for: {team1} vs {team2}")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            page = await context.new_page()
            
            # 1. Идем на страницу матчей
            await page.goto("https://www.hltv.org/matches", wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
            
            # 2. Ищем ссылку на матч по названиям команд (более гибкий поиск)
            match_link = await page.evaluate(f"""
                () => {{
                    const t1 = "{team1.lower()}";
                    const t2 = "{team2.lower()}";
                    
                    // Ищем во всех блоках матчей
                    const matchNodes = document.querySelectorAll('.upcomingMatch, .liveMatch, .match-day .match');
                    for (const node of matchNodes) {{
                        const text = node.innerText.toLowerCase();
                        // Проверяем наличие обеих команд в тексте блока
                        if (text.includes(t1) || text.includes(t2)) {{
                            // Если нашли хотя бы одну, проверяем вторую для надежности
                            if (text.includes(t1) && text.includes(t2)) {{
                                const a = node.querySelector('a');
                                return a ? a.href : null;
                            }}
                        }}
                    }}
                    return null;
                }}
            """)
            
            # 3. Если не нашли на главной, пробуем поиск через DuckDuckGo (более стабильно чем Google без API)
            if not match_link:
                search_query = f"{team1} vs {team2} hltv matches".replace(" ", "+")
                await page.goto(f"https://duckduckgo.com/?q={search_query}", wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                match_link = await page.evaluate("""
                    () => {
                        const links = Array.from(document.querySelectorAll('a'));
                        const hltvLink = links.find(a => a.href.includes('hltv.org/matches/'));
                        return hltvLink ? hltvLink.href : null;
                    }
                """)

            if not match_link:
                logger.warning(f"Match {team1} vs {team2} not found on HLTV")
                await browser.close()
                return None
            
            # 4. Переходим на страницу матча
            await page.goto(match_link, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)
            
            # 5. Извлекаем коэффициенты (несколько вариантов селекторов)
            odds = await page.evaluate("""
                () => {
                    const result = {};
                    
                    // Вариант 1: Стандартные ячейки
                    const oddsCells = document.querySelectorAll('.odds-cell');
                    if (oddsCells.length >= 2) {
                        result['team1'] = oddsCells[0].innerText.trim();
                        result['team2'] = oddsCells[1].innerText.trim();
                    }
                    
                    // Вариант 2: Блок букмекеров
                    if (!result.team1) {
                        const bookmakerOdds = document.querySelectorAll('.bookmaker-odds-container');
                        if (bookmakerOdds.length > 0) {
                            const oddsValues = bookmakerOdds[0].querySelectorAll('.odds-value');
                            if (oddsValues.length >= 2) {
                                result['team1'] = oddsValues[0].innerText.trim();
                                result['team2'] = oddsValues[1].innerText.trim();
                            }
                        }
                    }
                    
                    // Вариант 3: Сравнение коэффициентов
                    if (!result.team1) {
                        const externalOdds = document.querySelectorAll('.external-odds');
                        if (externalOdds.length >= 2) {
                            result['team1'] = externalOdds[0].innerText.trim();
                            result['team2'] = externalOdds[1].innerText.trim();
                        }
                    }

                    // Вариант 4: Блок сравнения коэффициентов (betting-listing)
                    if (!result.team1) {
                        const bettingListing = document.querySelector('.betting-listing');
                        if (bettingListing) {
                            const odds = bettingListing.querySelectorAll('.percentage');
                            if (odds.length >= 2) {
                                result['team1'] = odds[0].innerText.trim();
                                result['team2'] = odds[1].innerText.trim();
                            }
                        }
                    }
                    
                    return result;
                }
            """)
            
            await browser.close()
            
            if odds and 'team1' in odds and 'team2' in odds:
                try:
                    # Очищаем от лишних символов (оставляем только цифры и точку)
                    t1_odds_str = re.sub(r'[^0-9.]', '', odds['team1'])
                    t2_odds_str = re.sub(r'[^0-9.]', '', odds['team2'])
                    
                    if t1_odds_str and t2_odds_str:
                        return {"home_win": float(t1_odds_str), "away_win": float(t2_odds_str)}
                except:
                    pass
                
            return None
    except Exception as e:
        logger.error(f"Error scraping HLTV odds: {e}")
        return None

def get_hltv_odds(team1: str, team2: str):
    """Синхронная обертка для асинхронной функции."""
    try:
        # Создаем новый цикл событий, если текущий занят
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(get_hltv_odds_async(team1, team2))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Sync wrapper error: {e}")
        return None
