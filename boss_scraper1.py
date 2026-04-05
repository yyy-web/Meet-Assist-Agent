# ─────────────────────────────────────────────────────────────────────────────
#  boss_scraper.py  ·  BOSS 直聘爬虫（DrissionPage listen 版）
#
#  ⚠️  法律提示：本模块仅供个人学习研究，请遵守 BOSS 直聘用户协议。
#  建议：投递间隔 ≥ 8 秒，单日投递 ≤ 30 个。
#
#  依赖安装：pip install DrissionPage
#
#  使用方式：
#      python boss_scraper.py "Python后端工程师" 北京
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 城市代码映射 ──────────────────────────────────────────────
CITY_CODE: dict[str, str] = {
    "全国": "100010000", "北京": "101010100", "上海": "101020100",
    "广州": "101280100", "深圳": "101280600", "杭州": "101210100",
    "成都": "101270100", "武汉": "101200100", "南京": "101190100",
    "西安": "101110100", "重庆": "101040100", "苏州": "101190400",
    "天津": "101030100", "长沙": "101250100", "郑州": "101180100",
}

COOKIE_PATH = Path(__file__).parent / ".boss_cookies.json"


# ─────────────────────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class BossJob:
    job_id:       str  = ""
    title:        str  = ""
    company:      str  = ""
    location:     str  = ""
    salary:       str  = ""
    description:  str  = ""
    url:          str  = ""
    hr_name:      str  = ""
    hr_active:    str  = ""
    company_size: str  = ""
    experience:   str  = ""
    education:    str  = ""
    tags:         list = field(default_factory=list)
    apply_status: str  = "pending"
    apply_time:   str  = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def make_id(title: str, company: str) -> str:
        return hashlib.md5(f"{title}{company}".encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────
#  BossScraper 主类
# ─────────────────────────────────────────────────────────────

class BossScraper:
    """
    BOSS 直聘爬虫（DrissionPage listen 版）。

    核心思路：
      1. listen.start('joblist')  — 监听含 'joblist' 的 XHR/Fetch 请求
      2. page.get(url)            — 访问搜索页，自动触发 API 请求
      3. listen.wait(timeout=15)  — 阻塞等待响应包
      4. _parse_job_list(resp)    — 解析 zpData.jobList
      5. listen.stop()            — 停止监听，准备下一页
    """

    BASE_URL   = "https://www.zhipin.com"
    SEARCH_URL = "https://www.zhipin.com/web/geek/job"
    LOGIN_URL  = "https://www.zhipin.com/web/user/?ka=header-login"

    def __init__(
        self,
        headless:      bool  = False,
        request_delay: tuple = (2.0, 4.0),
        apply_delay:   tuple = (8.0, 15.0),
    ):
        self.headless      = headless
        self.request_delay = request_delay
        self.apply_delay   = apply_delay
        self.page          = None
        self._init_driver()

    # ── 初始化浏览器 ──────────────────────────────────────────

    def _init_driver(self):
        from DrissionPage import ChromiumPage, ChromiumOptions

        options = ChromiumOptions()
        if self.headless:
            options.headless()
        options.set_argument("--no-sandbox")
        options.set_argument("--disable-gpu")
        options.set_argument("--window-size=1440,900")
        options.set_argument("--lang=zh-CN,zh")
        options.set_argument("--log-level=3")

        self.page = ChromiumPage(options)
        logger.info("✅ DrissionPage 启动成功")

    # ── 登录管理 ──────────────────────────────────────────────

    def login_check(self, timeout: int = 180) -> bool:
        """
        登录流程：
          1. 访问首页建立 session
          2. 尝试加载本地 Cookie → 刷新验证
          3. Cookie 失效则跳转登录页等待扫码
          4. 登录成功后保存 Cookie
        """
        self.page.get(self.BASE_URL)
        self._delay(*self.request_delay)

        if COOKIE_PATH.exists():
            try:
                cookies = json.loads(COOKIE_PATH.read_text(encoding="utf-8"))
                for c in cookies:
                    self.page.set.cookies(c)
                self.page.refresh()
                self._delay(2, 3)
                if self._is_logged_in():
                    logger.info("✅ Cookie 登录成功")
                    return True
                logger.info("Cookie 已过期，需重新扫码")
            except Exception as e:
                logger.warning(f"Cookie 加载失败：{e}")

        if self.headless:
            logger.error("无头模式无法扫码，请设置 headless=False")
            return False

        logger.info("跳转登录页，请在浏览器中扫码...")
        self.page.get(self.LOGIN_URL)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_logged_in():
                self._save_cookies()
                logger.info("✅ 登录成功，Cookie 已保存")
                return True
            time.sleep(2)

        logger.error(f"登录超时（{timeout}秒）")
        return False

    def _is_logged_in(self) -> bool:
        for sel in [".nav-figure", ".user-nav", ".go-resume", ".user-menu"]:
            try:
                if self.page.ele(sel, timeout=1):
                    return True
            except Exception:
                pass
        try:
            names = {c.get("name", "") for c in self.page.cookies()}
            if names & {"bst", "wt2", "buid", "geek_zp_token"}:
                return True
        except Exception:
            pass
        url = self.page.url or ""
        return "geek" in url and "login" not in url

    def _save_cookies(self):
        try:
            COOKIE_PATH.write_text(
                json.dumps(list(self.page.cookies()), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"Cookie 已保存：{COOKIE_PATH}")
        except Exception as e:
            logger.warning(f"Cookie 保存失败：{e}")

    # ── 岗位搜索（listen 监听核心）────────────────────────────

    def search_jobs(
        self,
        keyword:    str,
        city:       str = "北京",
        max_pages:  int = 3,
        salary:     str = "",
        experience: str = "",
    ) -> list[BossJob]:
        """
        搜索岗位，基于 listen 监听 joblist API。

        每页流程：
          ① listen.start('joblist')  — 开始监听
          ② page.get(url)            — 触发 XHR
          ③ _wait_security_pass()    — 检测并等待安全验证页通过
          ④ listen.wait(timeout=15)  — 等待 joblist 响应
          ⑤ _parse_job_list(resp)    — 解析 zpData.jobList
          ⑥ listen.stop()            — 停止监听
        """
        from urllib.parse import urlencode

        city_code = CITY_CODE.get(city, "101010100")
        all_jobs: list[BossJob] = []

        for page_num in range(1, max_pages + 1):
            params = {"query": keyword, "city": city_code, "page": page_num}
            if salary:
                params["salary"] = salary
            if experience:
                params["experience"] = experience

            url = f"{self.SEARCH_URL}?{urlencode(params)}"
            logger.info(f"🔍 第 {page_num} 页：{url}")

            try:
                # ① 开始监听（匹配 URL 中含 'joblist' 的请求）
                self.page.listen.start("joblist")

                # ② 访问搜索页
                self.page.get(url)

                # ③ 检测安全验证页，若触发则等待自动通过后重新触发 joblist 请求
                if self._wait_security_pass(url):
                    # 安全验证通过后，页面已重新加载，joblist 请求已重新触发
                    # 重置监听，等待新的 joblist 响应
                    self.page.listen.stop()
                    self.page.listen.start("joblist")

                # ④ 阻塞等待 joblist XHR 响应
                resp = self.page.listen.wait(timeout=20)

                if resp is None:
                    logger.warning(f"  第 {page_num} 页监听超时，未捕获 joblist 响应")
                    break

                # ⑤ 解析 zpData.jobList
                page_jobs = self._parse_job_list(resp, city)

                if not page_jobs:
                    logger.info(f"  第 {page_num} 页无数据，停止翻页")
                    break

                all_jobs.extend(page_jobs)
                logger.info(f"  ✅ 本页获取 {len(page_jobs)} 个岗位")

                if page_num < max_pages:
                    self._delay(3.0, 6.0)

            except Exception as e:
                logger.warning(f"  第 {page_num} 页异常：{e}", exc_info=True)

            finally:
                # ⑥ 停止监听
                try:
                    self.page.listen.stop()
                except Exception:
                    pass

        logger.info(f"🎯 共获取 {len(all_jobs)} 个岗位")
        return all_jobs

    def _wait_security_pass(self, original_url: str, timeout: int = 60) -> bool:
        """
        检测是否被重定向到安全验证页（security.html）。

        BOSS 直聘安全验证页特征：
          - 页面标题含「请稍候」
          - URL 含 security.html / security1.html
          - 页面含「正在加载中」文字

        处理策略：
          - 安全验证页通常是前端 JS 自动完成验证后跳回原页，无需人工干预
          - 等待页面自动跳转回搜索页（URL 回到 zhipin.com/web/geek/job）
          - 若超时仍未跳转，记录警告并返回 False（调用方可选择中止或重试）

        返回值：
          True  — 检测到了安全验证页，且已等待其自动通过并跳回原页
          False — 未检测到安全验证页（正常情况）
        """
        SECURITY_SIGNS = ["请稍候", "security.html", "security1.html", "正在加载中"]

        # 短暂等待页面初始化
        time.sleep(1.5)

        current_url   = self.page.url or ""
        current_title = self.page.title or ""
        page_text     = self.page.html or ""

        is_security_page = (
            any(s in current_url   for s in ["security.html", "security1.html"])
            or "请稍候" in current_title
            or ("正在加载中" in page_text and "boss-loading" in page_text)
        )

        if not is_security_page:
            return False

        logger.warning("⚠️  检测到安全验证页，等待自动通过...")

        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(2)
            current_url = self.page.url or ""
            # 判断是否已跳回正常搜索页
            if "web/geek/job" in current_url and "security" not in current_url:
                logger.info("✅ 安全验证已通过，已跳回搜索页")
                self._delay(1.5, 2.5)  # 等待 joblist XHR 触发
                return True

        logger.warning(f"⏰ 安全验证等待超时（{timeout}s），当前 URL：{current_url}")
        return True  # 仍返回 True，让调用方重新等待 listen

    def _parse_job_list(self, resp, city: str) -> list[BossJob]:
        """
        解析 listen 捕获的 joblist API JSON 响应。

        确认的 JSON 结构：
            {
              "code": 0,
              "zpData": {
                "jobList": [
                  {
                    "encryptJobId":   "xxx",
                    "jobName":        "Python工程师",
                    "brandName":      "某公司",
                    "cityName":       "北京",
                    "salaryDesc":     "15-25K",
                    "jobLabels":      ["Python", "Django"],
                    "bossName":       "张HR",
                    "activeTimeDesc": "刚刚活跃",
                    "brandScaleName": "100-499人",
                    "experienceName": "3-5年",
                    "degreeName":     "本科"
                  },
                  ...
                ]
              }
            }
        """
        # —— 取响应体 ——
        try:
            body = resp.response.body           # DrissionPage ≥ 1.x 标准属性
        except AttributeError:
            body = getattr(resp, "body", None)  # 兼容旧版

        if not body:
            logger.warning("  响应体为空")
            return []

        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="ignore")

        # —— 解析 JSON ——
        try:
            data = body if isinstance(body, dict) else json.loads(body)
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON 解析失败：{e}")
            return []

        if data.get("code") != 0:
            logger.warning(
                f"  API 异常 code={data.get('code')} msg={data.get('message', '')}"
            )
            return []

        # —— 取 zpData.jobList ——
        job_list: list[dict] = data.get("zpData", {}).get("jobList", [])
        if not job_list:
            logger.info("  zpData.jobList 为空")
            return []

        # —— 逐条转换为 BossJob ——
        jobs: list[BossJob] = []
        for item in job_list:
            try:
                job_id  = item.get("encryptJobId", "")
                title   = item.get("jobName", "").strip()
                company = item.get("brandName", "").strip()

                if not title or not company:
                    continue

                url = (
                    f"https://www.zhipin.com/job_detail/{job_id}.html"
                    if job_id else ""
                )

                jobs.append(BossJob(
                    job_id       = job_id or BossJob.make_id(title, company),
                    title        = title,
                    company      = company,
                    location     = item.get("cityName", city).strip(),
                    salary       = item.get("salaryDesc", "").strip(),
                    url          = url,
                    hr_name      = item.get("bossName", "").strip(),
                    hr_active    = item.get("activeTimeDesc", "").strip(),
                    company_size = item.get("brandScaleName", "").strip(),
                    experience   = item.get("experienceName", "").strip(),
                    education    = item.get("degreeName", "").strip(),
                    tags         = item.get("jobLabels", []),
                ))
            except Exception as e:
                logger.debug(f"  解析单条 job 异常：{e}")
                continue

        return jobs

    # ── 岗位详情（XPath 补全 description）────────────────────

    def fetch_job_detail(self, job: BossJob) -> BossJob:
        """访问详情页，用 XPath 补全职位描述。"""
        if not job.url:
            return job
        try:
            from lxml import etree

            self.page.get(job.url)
            self._delay(*self.request_delay)

            html = etree.HTML(self.page.html)

            def _x(xp: str) -> str:
                r = html.xpath(xp)
                return r[0].strip() if r and isinstance(r[0], str) else ""

            job.description = _x(
                "//div[contains(@class,'job-detail-section')]"
                "//div[contains(@class,'text')]/text()"
            )
            if not job.description:
                paras = html.xpath("//div[contains(@class,'job-sec-text')]//text()")
                job.description = " ".join(p.strip() for p in paras if p.strip())

        except Exception as e:
            logger.debug(f"详情页获取失败（{job.title}）：{e}")
        return job

    def fetch_all_details(self, jobs: list[BossJob]) -> list[BossJob]:
        result = []
        for i, job in enumerate(jobs):
            logger.info(f"📄 详情 {i+1}/{len(jobs)}：{job.title} @ {job.company}")
            result.append(self.fetch_job_detail(job))
            if i < len(jobs) - 1:
                self._delay(2.0, 4.0)
        return result

    # ── 自动投递 ──────────────────────────────────────────────

    def apply_jobs(
        self,
        jobs:         list[BossJob],
        cover_letter: str = "",
        daily_limit:  int = 30,
    ) -> list[BossJob]:
        applied = 0
        results: list[BossJob] = []

        for job in jobs:
            if applied >= daily_limit:
                job.apply_status = "skipped_daily_limit"
                results.append(job)
                continue

            logger.info(f"📨 投递：{job.title} @ {job.company}")
            letter  = getattr(job, "cover_letter", "") or cover_letter
            success = self._apply_single(job, letter)

            if success:
                job.apply_status = "applied"
                job.apply_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                applied += 1
                logger.info("  ✅ 投递成功")
            else:
                logger.warning(f"  ❌ 投递失败：{job.apply_status}")

            results.append(job)
            self._delay(*self.apply_delay)

        logger.info(f"🎯 本次共投递 {applied} 个")
        return results

    def _apply_single(self, job: BossJob, cover_letter: str) -> bool:
        if not job.url:
            job.apply_status = "failed_no_url"
            return False
        try:
            self.page.get(job.url)
            self._delay(2.0, 3.5)

            btn = None
            for sel in [".btn-startchat", ".btn-primary.btn-chat",
                        "css:.op-btn-container .btn-startchat"]:
                try:
                    btn = self.page.ele(sel, timeout=5)
                    if btn:
                        break
                except Exception:
                    pass

            if not btn:
                job.apply_status = "failed_no_btn"
                return False

            btn.click()
            self._delay(1.5, 2.5)

            popup = self._handle_popup()
            if popup == "already_applied":
                job.apply_status = "already_applied"
                return True
            if popup in ("quota_exceeded", "login_required", "blocked"):
                job.apply_status = f"failed_{popup}"
                return False

            if cover_letter:
                input_el = None
                for sel in [".chat-input textarea", "#chat-input",
                            "tag:textarea", "css:[contenteditable='true']"]:
                    try:
                        input_el = self.page.ele(sel, timeout=4)
                        if input_el:
                            break
                    except Exception:
                        pass

                if input_el:
                    input_el.clear()
                    input_el.input(cover_letter[:500], by_js=False)
                    self._delay(0.8, 1.5)

                    sent = False
                    for sel in [".btn-send", ".send-btn", "css:button[type=submit]"]:
                        try:
                            send_btn = self.page.ele(sel, timeout=3)
                            if send_btn:
                                send_btn.click()
                                sent = True
                                break
                        except Exception:
                            pass
                    if not sent:
                        input_el.run_js("this.form && this.form.submit()")
                    self._delay(1.0, 2.0)

            job.apply_status = "applied"
            return True

        except Exception as e:
            logger.warning(f"  投递异常：{e}")
            job.apply_status = "failed_exception"
            return False

    def _handle_popup(self) -> str:
        time.sleep(1.0)
        text = self.page.html
        if "今日沟通人数已达上限" in text or "今日沟通次数" in text:
            return "quota_exceeded"
        if "已发送过求职申请" in text or "已经沟通过" in text:
            return "already_applied"
        if "请登录" in text or "登录后" in text:
            return "login_required"
        if "账号异常" in text or "频繁操作" in text:
            return "blocked"
        for sel in [".dialog-close", ".modal-close", ".btn-close"]:
            try:
                btn = self.page.ele(sel, timeout=1)
                if btn:
                    btn.click()
                    time.sleep(0.3)
            except Exception:
                pass
        return "ok"

    # ── 工具方法 ──────────────────────────────────────────────

    def _delay(self, min_sec: float, max_sec: float):
        time.sleep(random.uniform(min_sec, max_sec))

    def quit(self):
        if self.page:
            try:
                self.page.quit()
            except Exception:
                pass
            self.page = None
            logger.info("🔒 浏览器已关闭")


# ─────────────────────────────────────────────────────────────
#  便捷函数（供 auto_apply.py 调用）
# ─────────────────────────────────────────────────────────────

def scrape_boss_jobs(
    keyword:      str,
    city:         str  = "北京",
    max_pages:    int  = 2,
    fetch_detail: bool = True,
    headless:     bool = False,
) -> list[dict]:
    scraper = BossScraper(headless=headless)
    try:
        if not scraper.login_check():
            logger.error("登录失败，退出")
            return []

        jobs = scraper.search_jobs(keyword, city=city, max_pages=max_pages)

        if fetch_detail and jobs:
            jobs = scraper.fetch_all_details(jobs)

        return [{
            "job_id":       j.job_id,
            "title":        j.title,
            "company":      j.company,
            "location":     j.location,
            "salary":       j.salary,
            "description":  j.description or f"{j.title}，{j.company}，{'，'.join(j.tags)}",
            "url":          j.url,
            "score":        0.0,
            "cover_letter": "",
            "applied":      False,
        } for j in jobs]
    finally:
        scraper.quit()


# ─────────────────────────────────────────────────────────────
#  命令行入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    keyword = sys.argv[1] if len(sys.argv) > 1 else "Python后端工程师"
    city    = sys.argv[2] if len(sys.argv) > 2 else "北京"

    print(f"\n🔍 搜索：{keyword}（{city}）\n")
    jobs = scrape_boss_jobs(keyword, city=city, max_pages=1, headless=False)

    print(f"\n✅ 共获取 {len(jobs)} 个岗位：\n")
    for i, j in enumerate(jobs[:5], 1):
        print(f"{i}. [{j['salary']}] {j['title']} — {j['company']} ({j['location']})")
        print(f"   🔗 {j['url']}\n")