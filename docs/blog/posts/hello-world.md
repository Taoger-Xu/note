---
authors:
    - taoger
categories:
    - Lab
    - OS
date: 2023-12-01
nostatistics: true
---

# 操作系统 - lab2 线程调度

!!! abstract
    一些踩过的坑，以及一个 kernel 的运行流程总结

<!-- more -->

!!! info "基本信息"
    时间：2023 秋冬

    实验文档：[:octicons-book-16:](https://zju-sec.github.io/os23fall-stu/lab2/)

## kernel 执行流程


1.  也就是帮每个线程填好 task_struct
2.  对于这个被调度走的线程来说，他只是调用了 switch_to 函数，然后等着再返回罢了.虽然实际上 CPU 把此线程踢开去干别的活了，但本线程并不知道.所以 CPU 再次调度回此线程的时候，就该从 switch_to 函数返回，从而让该线程无感衔接。
3.  加载后我们的 sp 会设为之前分配页的顶端，ra 会设置为__dummy，这都是初始化时设计的


## 注意事项

注意到我们对于非 idle 线程第一次从 __switch_to 返回的都是 __dummy， 这是我们初始化的 PCB 被恢复出来的 ra 决定的

但是该线程第二次被调度的时候，它恢复的 PCB 就是它之前自己存下来的而不再是初始化的.这个存下来的 ra 就是 switch_to，__switch_to 函数运行时的 ra 当然是__switch_to 的调用者