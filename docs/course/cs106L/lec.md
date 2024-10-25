---
authors:
    - taoger
categories:
    - cpp
date: 2024-03-01
nostatistics: true
---

# CS106L

!!! abstract
    
<!-- more -->

!!! info cs106L 
    下面是完成CS106L中的assignment中学会的modern用法，不是课程的全部内容。

##  Cpp basic

### C++ 关键字

#### const

const用来修饰变量表示这个变量不允许被改变，

- 非const变量可以隐式地转化为const变量，**但是const变量不能被转化为非const变量**

const 成员函数，声明为 `const` 的成员函数中，`this` 的类型是 `const Complex *`；而如果没有声明为 `const`，则 `this` 的类型是 `Complex *`。

```c++
class Complex {
    // ...
    string toString() const;
    // ...
};

```



==good practice==：在自定义类中，如果要自定义一个容器，则给容器中的`iterator`类型，以及`begin()`和`end()`成员函数定义const版本和非const版本（所有的std容器都是这么做的，它们都有const版本的`iterator`、`begin()`、`end()`）。

```c++
using iterator = HashMapIterator<HashMap, false>;
using const_iterator = HashMapIterator<HashMap, true>;
iterator begin();
const_iterator begin() const;
```



#### explicit

在C++98中，最初的用途是防止单参数构造函数隐式地将一个值转换为该类的对象，比如下面的类：

```c++
class MyClass {
public:
    MyClass(int value) {
        // 构造函数的实现
    }
};
```

如果没有使用 `explicit` 关键字，那么可以隐式地将整数转换为 `MyClass` 类型的对象：

```c++
void function(MyClass obj) {
    // ...
}

function(5); // 隐式将 5 转换为 MyClass 类型

```

为了避免这种隐式转换，可以将构造函数声明为 `explicit`。使用 `explicit` 之后，你必须显式地调用构造函数来创建 `MyClass` 的对象。

```c++
class MyClass {
public:
    explicit MyClass(int value) {
        // 构造函数的实现
    }
};

function(5); // 错误：不能从 'int' 隐式转换为 'MyClass'
function(MyClass(5)); // 正确：显式转换
```

在C++11 标准中，`explicit` 关键字也可以用于防止类的转换运算符进行隐式转换。

```c++
class MyClass {
public:
    // ... 其他成员 ...

    operator bool() const {
        return true; // 假设总是返回 true
    }
};
```

如果没有 `explicit`，你可以在需要布尔值的上下文中隐式地使用 `MyClass` 对象

```c++
MyClass obj;
if (obj) {
    // ... obj 隐式转换为 bool
}
```



为了避免这种隐式转换，可以将转换运算符声明为 `explicit`

```c++
class MyClass {
public:
    // ... 其他成员 ...

    explicit operator bool() const {
        return true;
    }
};

MyClass obj;
if (obj) { // 错误：不能从 'MyClass' 隐式转换为 'bool'
}
if (static_cast<bool>(obj)) { // 正确：显式转换
    // ...
}
```

#### static_cast



#### auto

`auto`的具体用法如下：

- 提高代码可读性与维护性

  ```c++
  std::vector<std::string> words = {"hello", "world", "auto", "example"};
  std::map<std::string, std::vector<std::string>> dict;
  
  // 不使用 auto
  for (std::vector<std::string>::iterator it = words.begin(); it != words.end(); ++it) {
          std::cout << *it << " ";
  }
  	std::cout << std::endl;
  
  // 使用 auto
  for (auto it = words.begin(); it != words.end(); ++it) {
      std::cout << *it << " ";
  }
  	std::cout << std::endl;
  ```

- 模板类型函数的返回值：函数返回复杂的模板类型或者依赖于模板参数的类型时，使用 `auto` 可以自动处理类型推导。

  函数名称前面的`auto`不会做任何的类型推导工作。相反的，他只是暗示使用了C++11的**尾置返回类型**语法，即在函数形参列表后面使用一个”`->`“符号指出函数的返回类型。

  ```c++
  #include <vector>
  
  template<typename T>
  auto processVector(const std::vector<T>& vec) -> decltype(vec.front()) {
      // 假设处理后返回第一个元素
      return vec.front();
  }
  
  int main() {
      std::vector<double> numbers = {1.1, 2.2, 3.3};
      auto result = processVector(numbers);
      std::cout << "Result: " << result << std::endl;
  }
  ```

  在C++14标准下我们可以忽略尾置返回类型，只留下一个`auto`。使用这种声明形式，auto标示这里会发生类型推导，`decltype`说明`decltype`的规则将会被用到这个推导过程中

  ```c++
  template<typename Container, typename Index>    //C++14版本，
  decltype(auto)                                  //可以工作，
  authAndAccess(Container& c, Index i)            //但是还需要
  {                                               //改良
      authenticateUser();
      return c[i];
  }
  ```

- 泛型编程

  ```c++
  #include <iostream>
  #include <typeinfo>
  
  template<typename T, typename U>
  auto add(T t, U u) -> decltype(t + u) {
      return t + u;
  }
  
  int main() {
      auto sum = add(1, 2.5);  // 使用 auto 处理不同类型的加法
      std::cout << "Sum: " << sum << " Type: " << typeid(sum).name() << std::endl;
  }
  ```

`auto`的类型推导规则和模板类似：

- 类型说明符是一个指针或引用但不是通用引用

  ```c++
  auto x = 27;                    //情景三（x既不是指针也不是引用）
  ```

- 类型说明符一个通用引用

  ```c++
  auto&& uref1 = x;               //x是int左值，
                                  //所以uref1类型为int&
  auto&& uref2 = cx;              //cx是const int左值，
                                  //所以uref2类型为const int&
  auto&& uref3 = 27;              //27是int右值，
                                  //所以uref3类型为int&&
  ```

- 类型说明符既不是指针也不是引用

  ```c++
  const auto cx = x;              //情景三（cx也一样）
  const auto & rx=cx;             //情景一（rx是非通用引用）
  ```

## C++ 17

### std::optional

`std::optional` 是 C++17 标准库中引入的一个模板类，所以可以传入一个类型，从而将模版类进行实例化，例如：std::optional<int>`、`std::optional<myclass>

