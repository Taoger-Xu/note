# my cs note
https://taoger-xu.github.io/note/
## 按主题收录文章

首页的整站地图链接到本站「Topics」文章索引。新文章放在 `docs/` 下，在文件开头设置主题；构建时自动更新地图数量和主题列表，可同时指定多个主题：

```yaml
---
title: 我的量化阅读笔记
topics:
  - inference
  - architecture
---
```

主题 ID：`architecture`、`base-models`、`aigc`、`sft`、`dpo`、`rlhf`、`distillation`、`lora`、`post-training`、`inference`、`evaluation`、`reasoning`、`agent`、`harness`、`skills`、`rsi`、`training-systems`、`computer-systems`、`courses`、`blog`。主题定义位于 `hooks/topic_map.py`。

`docs/course/` 下的笔记自动归入课程，现有 CMU 15-213 和 MIT 6.S081 笔记同时归入计算机系统；其他未设置主题的文章会进入「其他文章」。新文章即使尚未加入 `mkdocs.yml` 导航，也能从主题索引找到。地图结构参考 [LLM Atlas](https://zhoujx4.github.io/llm-atlas/#整站地图)，链接指向本站内容。

Blog 保留在 `docs/blog/`，发布入口为 `/note/blog/`。博客文章会自动收录至地图中的「Blog 文章」；也可添加 `topics` 归入其他主题。索引使用 Blog 插件生成的正式文章地址，并排除归档、分类等重复列表。新博客文章继续设置 `date`、`authors` 等元信息，并用 `<!-- more -->` 分隔摘要与正文。
