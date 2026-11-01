"""Build a linked topic map and article archive from Markdown metadata."""
from html import escape
from pathlib import Path
import re

import yaml
from mkdocs.utils import get_relative_url
from mkdocs.plugins import event_priority

# id, title, short description, x, y, width, height
NODES = [
    ('architecture', '模型架构', 'Transformer · MoE · 多模态', 24, 70, 160, 64),
    ('base-models', '基座模型', 'Qwen · DeepSeek · Llama', 24, 252, 160, 64),
    ('aigc', 'AIGC 扩散', 'DiT · Flow · 视频', 24, 450, 160, 64),
    ('sft', 'SFT', '数据 · 模板 · Packing', 250, 252, 150, 64),
    ('dpo', 'DPO 系列', 'IPO · KTO · SimPO', 440, 152, 160, 64),
    ('rlhf', 'PPO · GRPO', 'DAPO · GSPO · CISPO', 440, 252, 160, 64),
    ('distillation', 'OPD / 蒸馏', '白盒 · 黑盒 · 自蒸馏', 440, 352, 160, 64),
    ('lora', 'LoRA 系列', 'QLoRA · DoRA · PiSSA — 可叠加在以上任一阶段', 250, 440, 350, 56),
    ('post-training', '对齐模型', '可用的模型', 680, 252, 150, 64),
    ('inference', '推理优化与解码', 'KV Cache · 量化 · 投机', 880, 120, 200, 64),
    ('evaluation', '评测', '基准 · judge · Rubric', 880, 230, 200, 64),
    ('reasoning', '推理模型', 'RLVR · 测试时扩展', 680, 400, 150, 64),
    ('agent', 'Agent 训练', 'Tool Use · Agentic RL', 880, 372, 200, 52),
    ('harness', 'Harness 工程', '执行循环 · 上下文 · 沙箱', 880, 440, 200, 52),
    ('skills', 'Skills', '能力封装与自迭代', 880, 508, 200, 52),
    ('rsi', 'RSI 自我改进', '', 680, 560, 150, 56),
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


def map_html():
    parts = ['<div class="topic-map-scroll" tabindex="0" role="region" aria-label="整站流程导览图，可横向滚动">',
             '<svg class="topic-map" viewBox="0 0 1110 650" role="img" aria-labelledby="topic-map-title topic-map-desc">',
             '<title id="topic-map-title">整站流程导览图</title>',
             '<desc id="topic-map-desc">从基座模型出发，经过后训练形成对齐模型，延伸至推理、评测和 Agent。</desc>',
             '<defs><marker id="topic-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 1 L9 5 L0 9 Z"/></marker></defs>',
             '<g class="topic-group topic-group--post"><rect x="228" y="112" width="392" height="428" rx="12"/><text x="246" y="142">后训练</text></g>',
             '<g class="topic-group topic-group--agent"><rect x="860" y="330" width="240" height="240" rx="12"/><text x="878" y="360">Agent</text></g>']
    paths = [
        ('M104 252 V134', True), ('M104 316 V450', True),
        ('M184 284 H250', False),
        ('M400 284 C420 284 420 184 440 184', False),
        ('M400 284 H440', False),
        ('M400 284 C420 284 420 384 440 384', False),
        ('M600 184 C640 184 640 268 680 268', False),
        ('M600 284 H680', False),
        ('M600 384 C640 384 640 300 680 300', False),
        ('M600 300 C648 318 652 388 680 426', False),
        ('M830 274 C856 274 856 152 880 152', False),
        ('M830 284 C856 284 856 262 880 262', False),
        ('M830 296 C856 296 856 398 880 398', False),
        ('M980 424 V440', False), ('M980 492 V508', False),
        ('M880 534 C852 534 848 588 830 588', True),
    ]
    for path, dashed in paths:
        style = ' topic-edge--dashed' if dashed else ''
        parts.append(f'<path class="topic-edge{style}" d="{path}" marker-end="url(#topic-arrow)"/>')
    labels = [(112, 200, 'start', '架构'), (112, 390, 'start', '另一条赛道'),
              (656, 350, 'start', '可验证奖励'), (855, 240, 'middle', '度量'),
              (838, 548, 'end', '自我改进')]
    for x, y, anchor, text in labels:
        parts.append(f'<text class="topic-edge-label" x="{x}" y="{y}" text-anchor="{anchor}">{text}</text>')
    for key, title, desc, x, y, width, height in NODES:
        if key in {'sft', 'dpo', 'rlhf', 'distillation', 'lora'}:
            group = 'post'
        elif key in {'inference', 'evaluation', 'reasoning'}:
            group = 'serve'
        elif key in {'agent', 'harness', 'skills'}:
            group = 'agent'
        elif key == 'post-training':
            group = 'hub'
        elif key == 'rsi':
            group = 'rsi'
        else:
            group = 'base'
        title_y = y + (24 if height == 52 else 30)
        if not desc:
            title_y = y + 34
        parts.append(
            f'<g class="topic-node topic-node--{group}">'
            f'<title>{escape(title)}</title>'
            f'<rect class="topic-node-card" x="{x}" y="{y}" width="{width}" height="{height}" rx="8"/>'
            f'<rect class="topic-node-bar" x="{x}" y="{y}" width="4" height="{height}"/>'
            f'<text class="topic-node-title" x="{x+18}" y="{title_y}">{escape(title)}</text>')
        if desc:
            fit = f' textLength="{width-30}" lengthAdjust="spacingAndGlyphs"' if key in {'architecture', 'base-models', 'sft', 'rlhf'} else ''
            parts.append(f'<text class="topic-node-desc" x="{x+18}" y="{title_y+20}"{fit}>{escape(desc)}</text>')
        parts.append('</g>')
    parts.append('</svg></div>')
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
        return markdown.replace('<!-- TOPIC_MAP -->', map_html())
    if page.file.src_uri == 'topics/index.md':
        return markdown.replace('<!-- TOPIC_ARCHIVE -->', archive_html(page, config))
    return markdown
