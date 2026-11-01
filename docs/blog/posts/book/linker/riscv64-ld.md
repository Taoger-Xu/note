---
authors:
    - taoger
categories:
    - ics
    - linker
date: 2024-10-24
nostatistics: true
---

!!! abstract
    本教程来自PLCT实验室的公开课教程[从零开始实现链接器](https://www.bilibili.com/video/BV1D8411j7fo/?spm_id_from=333.337.search-card.all.click&vd_source=e6c6915f1ab3ff5fc66616ddb7432e32)，我的实现仓库是[riscv64-ld](https://github.com/Taoger-Xu/riscv64-ld)

<!-- more -->

## 目标文件的格式

使用` objdump -h simpleObj.o`

![image-20240530192500158](assets/image-20240530192500158.png)

objdump 的 `-s` 参数可以将所有段的内容以十六进制的方式打印出来， `-d` 参数可以将所有包含指令的段反汇编



![image-20240530192726055](assets/image-20240530192726055.png)