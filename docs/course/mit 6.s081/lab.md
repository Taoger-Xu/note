---
authors:
    - taoger
categories:
    - cpp
date: 2024-05-26
nostatistics: true
---

# Lab: Data Lab

!!! abstract
    
<!-- more -->

!!! info xv6

## Lab0 实验环境搭建

### vscode 环境配置
[参考链接](https://sanbuphy.github.io/p/%E4%BC%98%E9%9B%85%E7%9A%84%E8%B0%83%E8%AF%95%E5%9C%A8vscode%E4%B8%8A%E5%AE%8C%E7%BE%8E%E8%B0%83%E8%AF%95xv6%E5%AE%8C%E7%BB%93/)

配置IntelliSense实现完美跳转

1. 安装 bear

2. 在xv6目录下执行`make clean && bear -- make qemu` ，将会生成一个 compile_commands.json

3. 将 compile_commands.json 移动到 .vscode目录下

4. 在.vscode目录下中为c_cpp_properties.json添加

运行用户程序

1. 在vscode终端下的DEBUG CONSOLE下输入

``` shell
# 这里的 _ls是需要调试的用户程序
-exec file ./user/_ls
```

```json
"compileCommands": "${workspaceFolder}/.vscode/compile_commands.json"
```

### 调试指南

### 常见问题

1. `qemu-system-riscv64: -gdb tcp::26000: Failed to find an available port: Address already in use`：

解放方法：
```shell
lsof -i :26000

kill -9 <PID>
```



## Chapter 3 Page tables

