"""
Flask 版文本词频分析 Web 应用 — 专为 Vercel Serverless 部署
入口文件: api/index.py
"""
from flask import Flask, request, render_template
import requests
import re
import os
import tempfile
from bs4 import BeautifulSoup
import jieba
from collections import Counter
from pyecharts.charts import WordCloud, Bar, Line, Pie, Funnel, Radar, Scatter, HeatMap
from pyecharts import options as opts
from pyecharts.globals import ThemeType

# 模板目录指向项目根目录下的 templates/（兼容本地 + Vercel 两种运行环境）
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"))

# ---- 内置停用词 ----
STOPWORDS = {
    "的", "了", "是", "在", "和", "及", "与", "等", "为", "对", "将", "有", "无", "不", "都", "可",
    "能", "会", "要", "让", "使", "被", "从", "到", "于", "以", "按", "经", "由", "共", "每", "各",
    "个", "件", "篇", "号", "年", "月", "日", "时", "分", "秒", "我", "你", "他", "她", "它", "我们",
    "你们", "他们", "这", "那", "此", "彼", "之", "其", "也", "还", "又", "但", "而", "或", "且", "则",
    "就", "却", "并", "虽", "然", "因", "为", "所", "以", "着", "过", "啊", "呀", "吗", "呢", "吧", "啦",
    "地", "得", "登录", "注册", "首页", "通知公告", "媒体聚焦", "比赛入口", "回到官网"
}

CHART_TYPES = [
    "词云",
    "柱状图（前20词频）",
    "折线图（前20词频）",
    "饼图（前20词频）",
    "漏斗图（前20词频）",
    "雷达图（前5词频）",
    "散点图（词长-词频）",
    "热力图（前10词频）"
]


# ============================================================
# 核心功能函数（与 Streamlit 版逻辑一致）
# ============================================================

def fetch_text_from_url(url):
    """从 URL 抓取并清洗文本内容"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Referer": url,
            "Accept-Language": "zh-CN,zh;q=0.9"
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None, f"请求失败，状态码：{response.status_code}"

        # 编码自动检测
        raw_content = response.content
        meta_match = re.search(
            rb'<meta[^>]+charset=["\']?([a-zA-Z0-9\-_]+)',
            raw_content[:4096], re.IGNORECASE
        )
        try:
            import chardet
            detected = chardet.detect(raw_content)
            detected_enc = detected.get("encoding") if detected else None
        except ImportError:
            detected_enc = None

        encoding = (
            (meta_match.group(1).decode("ascii") if meta_match else None)
            or detected_enc
            or response.apparent_encoding
            or "utf-8"
        )
        if encoding and encoding.lower() in ("gb2312", "gb18030"):
            encoding = "gbk"

        html_text = raw_content.decode(encoding, errors="replace")
        soup = BeautifulSoup(html_text, "html.parser")

        # 移除 JS/CSS/导航/页脚
        for tag in soup.find_all(["script", "style", "noscript", "iframe", "nav", "footer"]):
            tag.decompose()

        if soup.body is None:
            return None, "页面中未找到 body 标签，无法提取正文"

        # 精准提取正文容器
        content_selectors = [
            {"class_": "content"}, {"class_": "article-content"},
            {"class_": "article"}, {"id": "content"},
            {"id": "article"}, {"class_": "main-content"},
            {"class_": "post-content"},
        ]
        article_tag = None
        for sel in content_selectors:
            article_tag = soup.body.find("div", sel) or soup.body.find("article", sel)
            if article_tag:
                break

        if article_tag:
            text = article_tag.get_text(separator=" ")
        else:
            text = soup.body.get_text(separator=" ")

        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text, None
    except Exception as e:
        return None, f"抓取失败：{str(e)}"


def segment_and_count(text):
    """jieba 分词 + 词频统计"""
    if not text:
        return None, "文本为空"
    words = jieba.lcut(text, cut_all=False)
    filtered_words = [
        word for word in words
        if len(word) >= 2 and word not in STOPWORDS and not word.isdigit()
    ]
    if not filtered_words:
        return None, "分词后无有效词汇"
    return Counter(filtered_words), None


def filter_low_freq(word_freq, min_freq):
    """过滤低频词"""
    return {word: freq for word, freq in word_freq.items() if freq >= min_freq}


def generate_chart(chart_type, word_freq):
    """生成 pyecharts 图表 HTML"""
    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top20 = sorted_freq[:20]
    top5 = sorted_freq[:5]
    top10 = sorted_freq[:10]

    if chart_type == "词云":
        chart = (
            WordCloud(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="600px"))
            .add("", sorted_freq, word_size_range=[20, 100])
            .set_global_opts(title_opts=opts.TitleOpts(title="文本词云图"))
        )
    elif chart_type == "柱状图（前20词频）":
        chart = (
            Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="600px"))
            .add_xaxis([w for w, _ in top20])
            .add_yaxis("词频", [f for _, f in top20])
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频前20柱状图"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45))
            )
        )
    elif chart_type == "折线图（前20词频）":
        chart = (
            Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="600px"))
            .add_xaxis([w for w, _ in top20])
            .add_yaxis("词频", [f for _, f in top20],
                       markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频前20折线图"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45))
            )
        )
    elif chart_type == "饼图（前20词频）":
        chart = (
            Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="600px"))
            .add("", top20)
            .set_global_opts(title_opts=opts.TitleOpts(title="词频前20饼图"))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        )
    elif chart_type == "漏斗图（前20词频）":
        chart = (
            Funnel(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="600px"))
            .add("词频", top20,
                 label_opts=opts.LabelOpts(position="inside", formatter="{b}: {c}"))
            .set_global_opts(title_opts=opts.TitleOpts(title="词频前20漏斗图"))
        )
    elif chart_type == "雷达图（前5词频）":
        if len(top5) < 3:
            return None, "词频前5词汇不足，无法生成雷达图"
        chart = (
            Radar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="600px"))
            .add_schema(
                schema=[opts.RadarIndicatorItem(name=w, max_=f + 5) for w, f in top5],
                splitarea_opt=opts.SplitAreaOpts(is_show=True)
            )
            .add("词频", [[f for _, f in top5]],
                 areastyle_opts=opts.AreaStyleOpts(opacity=0.2))
            .set_global_opts(title_opts=opts.TitleOpts(title="词频前5雷达图"))
        )
    elif chart_type == "散点图（词长-词频）":
        chart = (
            Scatter(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="600px"))
            .add_xaxis([len(w) for w, _ in sorted_freq[:50]])
            .add_yaxis("词频", [f for _, f in sorted_freq[:50]],
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词长度与词频散点图"),
                xaxis_opts=opts.AxisOpts(name="词长度", min_=2),
                yaxis_opts=opts.AxisOpts(name="词频")
            )
        )
    elif chart_type == "热力图（前10词频）":
        if not top10:
            return None, "词频前10词汇不足，无法生成热力图"
        max_freq = max(f for _, f in top10)
        chart = (
            HeatMap(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="600px"))
            .add_xaxis([w for w, _ in top10])
            .add_yaxis("词频", ["词频值"], [[i, 0, f] for i, (_, f) in enumerate(top10)])
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频前10热力图"),
                visualmap_opts=opts.VisualMapOpts(max_=max_freq, is_piecewise=True)
            )
        )
    else:
        return None, "不支持的图表类型"

    try:
        return chart.render_embed(), None
    except Exception:
        fd, tmp_path = tempfile.mkstemp(suffix=".html", prefix="chart_")
        os.close(fd)
        chart.render(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp_path)
        return html, None


# ============================================================
# Flask 路由
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    info = None
    top20 = None
    chart_html = None
    url = ""
    min_freq = 2
    chart_type = "词云"

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        try:
            min_freq = int(request.form.get("min_freq", "2"))
        except ValueError:
            min_freq = 2
        chart_type = request.form.get("chart_type", "词云")

        if not url:
            error = "请输入文章 URL！"
        else:
            # 抓取
            text, err = fetch_text_from_url(url)
            if err:
                error = err
            elif text:
                # 分词 + 词频统计
                word_freq, err = segment_and_count(text)
                if err:
                    error = err
                else:
                    # Top20（基于原始数据）
                    top20 = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

                    # 低频过滤
                    filtered_freq = filter_low_freq(word_freq, min_freq)
                    if not filtered_freq:
                        info = f"当前阈值下无词频≥{min_freq}的词汇，请调低阈值"
                    else:
                        # 生成图表
                        result, err = generate_chart(chart_type, filtered_freq)
                        if err:
                            error = err
                        else:
                            chart_html = result

    return render_template(
        "index.html",
        error=error,
        info=info,
        top20=top20,
        chart_html=chart_html,
        url=url,
        min_freq=min_freq,
        chart_type=chart_type,
        chart_types=CHART_TYPES,
        enumerate=enumerate,
    )
