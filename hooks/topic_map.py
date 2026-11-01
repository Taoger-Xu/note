"""Build a linked topic map and article archive from Markdown metadata."""
from html import escape
from pathlib import Path
import re

import yaml
from mkdocs.utils import get_relative_url
from mkdocs.plugins import event_priority

# id, title, short description, x, y, width, height
NODES = [
    ('architecture', '模型架构', 'Attention / MoE / 多模态', 24, 65, 180, 74),
    ('base-models', '基座模型', '预训练 / 模型家族', 24, 255, 180, 74),
    ('aigc', 'AIGC', '扩散 / 图像与视频生成', 24, 465, 180, 74),
    ('sft', 'SFT 监督微调', '指令数据 / 训练目标', 260, 255, 180, 74),
    ('dpo', 'DPO 系列', '离线偏好优化', 480, 145, 180, 74),
    ('rlhf', 'PPO / GRPO', '奖励与强化学习', 480, 255, 180, 74),
    ('distillation', 'OPD / 蒸馏', '教师与学生模型', 480, 365, 180, 74),
    ('lora', 'LoRA 系列', '参数高效微调 · 可用于不同训练阶段', 260, 470, 400, 74),
    ('post-training', '对齐与后训练', '能力与行为优化', 720, 255, 180, 74),
    ('inference', '推理优化与解码', '量化 / KV Cache / 投机解码', 960, 65, 220, 74),
    ('evaluation', '评测', '基准 / 可靠性 / 模型评价', 960, 185, 220, 74),
    ('reasoning', '推理模型', 'RLVR / 测试时计算', 720, 420, 180, 74),
    ('agent', 'Agent 训练', '工具使用 / Agentic RL', 960, 345, 220, 74),
    ('harness', 'Harness 工程', '上下文 / 执行循环 / 沙箱', 960, 445, 220, 74),
    ('skills', 'Skills', '技能组织与复用', 960, 545, 220, 74),
    ('rsi', 'RSI 自我改进', '反馈与迭代', 720, 570, 180, 74),
]
EXTRA_TOPICS = [
    ('training-systems', '训练系统 / 分布式', '并行训练、显存与计算效率'),
    ('computer-systems', '计算机系统', '操作系统、体系结构与 GPU'),
    ('courses', '课程笔记', '课程讲义、实验与项目'),
    ('blog', 'Blog 文章', '博客随笔与技术记录'),
    ('other', '其他文章', '尚未设置主题的文章'),
]
TITLES = {n[0]: n[1] for n in NODES + EXTRA_TOPICS}
ARTICLES = {}


@event_priority(-100)
def on_files(files, config):
    """Rebuild the index on every build, including mkdocs serve rebuilds."""
    global ARTICLES
    ARTICLES = {key: [] for key in TITLES}
    # Material rewrites post URLs at priority -50 and excludes posts from
    # ordinary navigation. Index its published posts after those URLs settle.
    blog = config.plugins.get('material/blog') or config.plugins.get('blog')
    published_posts = {post.file.src_uri for post in blog.blog.posts} if blog else set()
    for file in files.documentation_pages(inclusion=lambda _: True):
        if file.src_uri.startswith('blog/') and file.src_uri not in published_posts:
            continue
        if file.src_uri in ('index.md', 'topics/index.md'):
            continue
        source = Path(file.abs_src_path).read_text(encoding='utf-8-sig')
        match = re.match(r'^---\s*\n(.*?)\n---\s*(?:\n|$)', source, re.S)
        meta = (yaml.safe_load(match[1]) or {}) if match else {}
        body = source[match.end():] if match else source
        heading = re.search(r'^#\s+(.+)', body, re.M)
        title = str(meta.get('title') or (heading[1].strip() if heading else Path(file.src_uri).stem))
        if file.src_uri.startswith('course/') and file.src_uri != 'course/index.md':
            title = f'{Path(file.src_uri).parent.name} · {title}'
        topics = meta.get('topics', [])
        if isinstance(topics, str):
            topics = [topics]
        if not isinstance(topics, list):
            raise ValueError(f'{file.src_uri}: topics must be a list or a topic id')
        if file.src_uri in published_posts:
            topics = topics + ['blog']
            categories = {str(category).lower() for category in meta.get('categories', [])}
            if categories & {'ics', 'cpp', 'modern cpp', 'linker', 'os', 'c', 'cuda', 'hpca', 'ysyx'}:
                topics = topics + ['computer-systems']
        if file.src_uri.startswith(('course/', 'blog/posts/course/')):
            topics = topics + ['courses']
        if file.src_uri.startswith(('course/cmu15-213/', 'course/mit 6.s081/')):
            topics = topics + ['computer-systems']
        if not topics:
            topics = ['other']
        for topic in set(topics):
            if topic not in ARTICLES:
                raise ValueError(f'{file.src_uri}: unknown topic {topic!r}; see hooks/topic_map.py')
            ARTICLES[topic].append((title, file.url))
    return files


def topic_link(topic, page, config):
    archive = 'topics/' if config['use_directory_urls'] else 'topics/index.html'
    return escape(get_relative_url(archive, page.file.url) + '#' + topic, quote=True)


def map_html(page, config):
    parts = ['<div class="topic-map-scroll" tabindex="0" role="region" aria-label="整站地图，可横向滚动">',
             '<svg class="topic-map" viewBox="0 0 1204 680" role="group" aria-labelledby="topic-map-title topic-map-desc">',
             '<title id="topic-map-title">文章主题地图</title>',
             '<desc id="topic-map-desc">基座模型经过后训练走向推理、评测与 Agent。每个主题链接到本站文章索引。</desc>',
             '<defs><marker id="topic-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 L9 5 L1 9"/></marker></defs>',
             '<rect class="topic-panel" x="240" y="105" width="440" height="460" rx="16"/>',
             '<text class="topic-panel-label" x="260" y="130">后训练</text>',
             '<rect class="topic-panel" x="940" y="305" width="260" height="335" rx="16"/>',
             '<text class="topic-panel-label" x="960" y="330">Agent</text>']
    paths = [
        'M114 255 V139', 'M114 329 V465', 'M204 292 H260',
        'M440 292 C460 292 460 182 480 182', 'M440 292 H480',
        'M440 292 C460 292 460 402 480 402',
        'M660 182 C690 182 690 280 720 280', 'M660 292 H720',
        'M660 402 C690 402 690 304 720 304',
        'M660 329 C690 350 690 420 720 445',
        'M900 278 C930 278 930 102 960 102',
        'M900 292 C930 292 930 222 960 222',
        'M900 306 C930 306 930 382 960 382',
        'M1070 419 V445', 'M1070 519 V545',
        'M960 582 C930 582 930 607 900 607',
    ]
    parts.extend(f'<path class="topic-edge" d="{d}" marker-end="url(#topic-arrow)"/>' for d in paths)
    for key, title, desc, x, y, width, height in NODES:
        count = len(ARTICLES[key])
        status = f'{count} 篇' if count else '待更新'
        parts.append(
            f'<a class="topic-node" href="{topic_link(key, page, config)}" aria-label="{escape(title)}，{status}">'
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="9"/>'
            f'<text class="topic-node-title" x="{x+13}" y="{y+25}">{escape(title)}</text>'
            f'<text class="topic-node-desc" x="{x+13}" y="{y+46}">{escape(desc)}</text>'
            f'<text class="topic-node-count" x="{x+13}" y="{y+64}">{status} →</text></a>')
    parts.append('</svg></div><div class="topic-map-extras">')
    for key, title, _ in EXTRA_TOPICS:
        if key == 'other' and not ARTICLES[key]:
            continue
        parts.append(f'<a href="{topic_link(key, page, config)}">{escape(title)} <span>{len(ARTICLES[key])} 篇 →</span></a>')
    parts.append('</div>')
    return ''.join(parts)


def archive_html(page, config):
    parts = ['<div class="topic-archive">']
    for key, title in TITLES.items():
        entries = ARTICLES[key]
        parts.append(f'<section class="topic-archive-section" aria-labelledby="{key}"><h2 id="{key}">{escape(title)} <span>{len(entries)} 篇</span></h2>')
        if entries:
            parts.append('<ul>')
            for title, url in sorted(entries):
                href = escape(get_relative_url(url, page.file.url), quote=True)
                parts.append(f'<li><a href="{href}">{escape(title)}</a></li>')
            parts.append('</ul>')
        else:
            parts.append('<p class="topic-empty">这个主题还没有文章，后续笔记将在这里收录。</p>')
        parts.append('<a class="topic-back" href="' + escape(get_relative_url('', page.file.url), quote=True) + '#site-map">返回整站地图 ↑</a></section>')
    parts.append('</div>')
    return ''.join(parts)


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri == 'index.md':
        return markdown.replace('<!-- TOPIC_MAP -->', map_html(page, config))
    if page.file.src_uri == 'topics/index.md':
        return markdown.replace('<!-- TOPIC_ARCHIVE -->', archive_html(page, config))
    return markdown
