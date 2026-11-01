# my cs note
https://taoger-xu.github.io/note/
## 按主题收录文章

首页直接展示静态流程导览图，图块暂不提供跳转。Topics 与 Paper 导航入口已隐藏；原有文章与主题索引代码保留，供后续优化使用。

如后续启用主题索引，新文章可放在 `docs/` 下，在文件开头设置主题，可同时指定多个主题：

```yaml
---
title: 我的量化阅读笔记
topics:
  - inference
  - architecture
---
```

主题 ID：`architecture`、`base-models`、`aigc`、`sft`、`dpo`、`rlhf`、`distillation`、`lora`、`post-training`、`inference`、`evaluation`、`reasoning`、`agent`、`harness`、`skills`、`rsi`、`training-systems`、`computer-systems`、`courses`、`blog`。主题定义位于 `hooks/topic_map.py`。

`docs/course/` 仅保留 `cs336/` 与 `cmu-agent/` 两门课程，笔记自动归入课程；其他未设置主题的文章会进入「其他文章」。新文章即使尚未加入 `mkdocs.yml` 导航，也能从主题索引找到。地图结构参考 [LLM Atlas](https://zhoujx4.github.io/llm-atlas/#整站地图)，目前仅作静态导览展示。

Blog 保留在 `docs/blog/`，发布入口为 `/note/blog/`。Blog 正常展示文章；预留主题索引仍收录博客文章，也可添加 `topics` 归入其他主题。索引使用 Blog 插件生成的正式文章地址，并排除归档、分类等重复列表。新博客文章继续设置 `date`、`authors` 等元信息，并用 `<!-- more -->` 分隔摘要与正文。

## 主题目录

旧博客文章和配图已清空，`docs/blog/posts/` 保留为空目录。Blog 入口保留，后续可继续发布新文章。

后续主题文章放在 `docs/topics/` 对应子目录下：`model`（基座模型）、`architecture`、`aigc`、`sft`、`dpo`、`rlhf`、`distillation`、`lora`、`post-training`、`inference`、`evaluation`、`reasoning`、`agent`、`harness`、`skills`、`rsi`、`training-systems`。空目录通过 `.gitkeep` 保留在 Git 中，不生成占位文章。`model` 目录内的文章可使用 `topics: [base-models]` 标记主题。
