---
authors:
    - taoger
categories:
    - Cpp
date: 2024-12-11
nostatistics: true
---

# C++模板

!!! abstract
    hello

<!-- more -->

!!! info hello
   hello

   [test](https://zju-sec.github.io/os23fall-stu/lab2/)

## 函数模板

函数模板不是函数，只有实例化[[2\]](https://mq-b.github.io/Modern-Cpp-templates-tutorial/md/第一部分-基础知识/01函数模板#fn2)函数模板，编译器才能生成实际的函数定义。不过在很多时候，它看起来就像普通函数一样

### 定义模板

```cpp
template<typename T>
T max(T a,T b){
    return a > b ? a : b;
}
```



### 可变参数模板

可变参数模板是 C++11 引入的一项强大特性，使得函数模板和类模板能够接受任意数量的模板参数。

对于下面的需求：

实现函数 sum，支持 sum(1,2,3.5,x,n...) 即函数 sum 支持任意类型，任意个数的参数进行调用，你应该如何实现？

```cpp

// Args... 是一个模板参数包（parameter pack），它表示零个或多个类型参数
template<typename...Args>

// args... 是对应的函数参数包，表示传入的函数参数
void sum(Args...args){}
```



## 模板类型推导

