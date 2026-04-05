# test_boss_login.py  ·  BOSS 直聘登录 + 搜索测试（DrissionPage 版）
# 运行：python test_boss_login.py

import json
import logging
import random
import re
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

COOKIE_PATH = Path(__file__).parent / ".boss_cookies.json"


def get_page():
    """初始化 DrissionPage，自动处理反检测和驱动版本。"""
    from DrissionPage import ChromiumPage, ChromiumOptions

    options = ChromiumOptions()
    options.set_argument("--no-sandbox")
    options.set_argument("--disable-gpu")
    options.set_argument("--window-size=1440,900")
    options.set_argument("--lang=zh-CN,zh")
    options.set_argument("--log-level=3")

    page = ChromiumPage(options)
    logger.info("✅ DrissionPage 启动成功")
    return page


def is_logged_in(page) -> bool:
    for sel in [".nav-figure", ".user-nav", ".go-resume", ".user-menu"]:
        try:
            el = page.ele(sel, timeout=1)
            if el:
                return True
        except Exception:
            pass
    try:
        names = {c.get("name", "") for c in page.cookies()}
        if any(k in names for k in ("bst", "wt2", "buid", "geek_zp_token")):
            return True
    except Exception:
        pass
    url = page.url or ""
    if "geek" in url and "login" not in url and "user" not in url:
        return True
    return False


def load_cookies(page) -> bool:
    if not COOKIE_PATH.exists():
        return False
    try:
        with open(COOKIE_PATH, "r", encoding="utf-8") as f:
            cookies = json.load(f)
        for c in cookies:
            page.set.cookies(c)
        page.refresh()
        time.sleep(3)
        return is_logged_in(page)
    except Exception as e:
        logger.warning(f"Cookie 加载失败：{e}")
        return False


def save_cookies(page):
    try:
        cookies = list(page.cookies())
        with open(COOKIE_PATH, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Cookie 已保存（{len(cookies)} 条）-> {COOKIE_PATH}")
    except Exception as e:
        logger.warning(f"Cookie 保存失败：{e}")


def scroll_to_bottom(page):
    total_h = page.run_js("return document.body.scrollHeight")
    current = 0
    while current < total_h:
        current += 400
        page.run_js(f"window.scrollTo(0, {current});")
        time.sleep(random.uniform(0.15, 0.3))
    page.run_js("window.scrollTo(0, 0);")
    time.sleep(0.5)


def main():
    page = get_page()

    try:
        # Step 1: 访问首页
        logger.info("访问 BOSS 直聘首页...")
        page.get("https://www.zhipin.com")
        time.sleep(4)
        logger.info(f"页面标题：{page.title}")

        # Step 2: Cookie 登录
        logger.info("尝试 Cookie 登录...")
        if load_cookies(page):
            logger.info("✅ Cookie 登录成功")
        else:
            # Step 3: 扫码登录
            logger.info("跳转登录页，请扫码...")
            page.get("https://www.zhipin.com/web/user/?ka=header-login")
            time.sleep(2)
            logger.info(f"登录页标题：{page.title}")
            logger.info("⏳ 等待扫码（最多 180 秒）...")

            start = time.time()
            logged = False
            while time.time() - start < 180:
                if is_logged_in(page):
                    logged = True
                    break
                time.sleep(2)

            if logged:
                save_cookies(page)
                logger.info("✅ 扫码登录成功")
            else:
                logger.error("❌ 登录超时")
                page.quit()
                return

        # Step 4: 测试搜索
        logger.info("\n测试岗位搜索页...")
        page.get(
            "https://www.zhipin.com/web/geek/job"
            "?query=Python%E5%90%8E%E7%AB%AF%E5%B7%A5%E7%A8%8B%E5%B8%88"
            "&city=101010100&page=1"
        )
        time.sleep(5)
        logger.info(f"搜索页标题：{page.title}")

        logger.info("滚动页面触发懒加载...")
        scroll_to_bottom(page)

        # lxml XPath 解析
        from lxml import etree
        html  = etree.HTML(page.html)
        cards = []
        for xp in [
            "//li[contains(@class,'job-card-wrapper')]",
            "//div[contains(@class,'job-card-wrapper')]",
        ]:
            cards = html.xpath(xp)
            if cards:
                break

        logger.info(f"\n✅ 找到岗位卡片：{len(cards)} 个\n")

        if cards:
            for i, card in enumerate(cards[:5], 1):
                def _x(xp, default=""):
                    r = card.xpath(xp)
                    return r[0].strip() if r and isinstance(r[0], str) else default
                title    = _x(".//span[@class='job-name']/text()")
                salary   = _x(".//span[@class='salary']/text()")
                company  = (
                    _x(".//h3[@class='company-name']/a/text()") or
                    _x(".//h3[@class='company-name']/text()")
                )
                location = _x(".//span[@class='job-area']/text()")
                tags     = [t.strip() for t in card.xpath(".//ul[@class='tag-list']/li/text()") if t.strip()]
                print(f"  {i}. [{salary}] {title}")
                print(f"     公司：{company}  地点：{location}")
                print(f"     标签：{' | '.join(tags)}\n")
        else:
            logger.warning("未找到岗位卡片，保存调试文件...")
            page.get_screenshot(path=str(Path(__file__).parent), name="debug_search.png")
            with open("debug_search.html", "w", encoding="utf-8") as f:
                f.write(page.html)
            logger.info("截图已保存：debug_search.png")
            logger.info("源码已保存：debug_search.html（用浏览器打开查看实际结构）")

        input("\n按 Enter 关闭浏览器...")

    finally:
        page.quit()
        logger.info("浏览器已关闭")


if __name__ == "__main__":
    main()