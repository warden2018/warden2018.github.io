---
layout: post
title:  投影矩阵和最小二乘
date:   2020-01-30
categories: Linear Algebra
author: Yang
tags: [Least Squares, Linear Algebra,Projection Matrix]
comments: true
toc: true
mathjax : true
---
# 投影矩阵
之前的一篇博客讲到了矩阵的四个子空间：行空间，列空间，零空间和左零空间。每一个空间唯一性地由这个矩阵决定，因为这个矩阵的列可能线性无关，可能线性相关，如果线性相关，能够在其中找到多少个线性无关的向量。所以，不同的矩阵所形成的对应的四个子空间情况各不相同。我们为了能够理解给定的矩阵四个空间分布情况，就引入了投影矩阵的概念，用来分析在空间中任意向量到这个矩阵的“距离”。下面我们具体看一下：
首先回顾一下矩阵的列空间$C(A)$和对应的左零空间$N(A^T)$，它们所具有的最优秀的属性就是**垂直**。
矩阵$A$的投影矩阵

$$
P=A(AA^T)^{-1}A^T
$$

给定任意一个与$A$列相同维度的向量 $ \boldsymbol b$，它与这个矩阵列空间的关系是什么？我们怎么能够几何直观地理解一下？

$$
A=\begin{bmatrix}1&4\\ 2&5\\\ 3&6\end{bmatrix}
$$

![列向量子空间](/images/projectionMatrix.png)

两个列向量$\boldsymbol u=\begin{bmatrix}1\\ 2\\ 3\end{bmatrix},\boldsymbol v=\begin{bmatrix}4\\ 5\\ 6\end{bmatrix}$构成列空间（一个平面）。
![列向量子空间](/images/projectionMatrix2.png)
$\boldsymbol {per}=\begin{bmatrix}1\\\ -2\\ 1\end{bmatrix}$是左零空间的一个向量。$\boldsymbol b$是三维空间内的任意向量，我们举出的是一般性的例子，即这个向量不在这两个子空间的任何一个空间内。
1. 如果在列空间，那么投影的作用是保留全部--即该向量保持不变
2. 如果在左零空间，那么投影的结果是0

![列向量子空间](/images/projectionMatrix3.png)
那么，投影矩阵对该任意向量的作用就是将这个向量对列空间投影，得到$\boldsymbol{projection}$向量，称为投影向量，再求一次向量减法运算，得到$\boldsymbol{e}$，误差向量，代表了这个任意给定向量和列空间的差别。

$$
\boldsymbol{projection} + \boldsymbol{e}=\boldsymbol{b}
$$

## 投影矩阵的性质
1. 幂等性（idempotent）：$P=P^2$。这也是投影矩阵的定义。由上面的描述我们可以得到任意向量做一次投影之后的结果已经在原矩阵的列空间内，继续做投影依旧保持第一次的投影不变。


# 最小二乘

## 问题描述

已知空间当中的一些点，如何找到一条直线，能够尽可能地离所有点最近？

## 问题梳理

![列向量子空间](/images/Least_Square.png)
假设二维空间有$n$个点，我们定义点$x=\begin{bmatrix}x_i\\ y_i \end{bmatrix}$到直线$y=ax+b$的偏差为$e=ax_i+b-y_i$，最小化这些偏差的平方

$$
S=\sqrt{\sum_{i=0}^n {(e_i)^2}}
$$

为了最小化这个式子，根据式子的特点，我们可以将问题转化成为通过寻找向量的模长的最小值解决。这个向量是：

$$
\boldsymbol {e}=\begin{bmatrix} e_1& e_e  \cdots & e_n     \end{bmatrix}^T
$$

$$
\boldsymbol{e}=\begin{bmatrix} e_1\\\  \vdots \\\ e_n    \end{bmatrix}=\begin{bmatrix} y_1-b-ax_1\\\  \vdots y_n-b-ax_n    \end{bmatrix}=\begin{bmatrix} y_1\\\  \vdots y_n   \end{bmatrix}-a\begin{bmatrix} x_1\\\  \vdots \\ x_n   \end{bmatrix}-b\begin{bmatrix} 1\\\  \vdots \\\1   \end{bmatrix}=\begin{bmatrix} y_1\\\  \vdots \\ y_n   \end{bmatrix}-\begin{bmatrix} 1&x_1\\\  \vdots &\vdots\\\1 &x_n  \end{bmatrix}\begin{bmatrix}b\\\a\end{bmatrix}=\boldsymbol{y}-M\boldsymbol{x}
$$

## 投影已知向量

最小化该向量的长度，就是尽可能地在矩阵$M$的列空间中，找到一个最接近于$y$的向量，那么，直观理解就是该向量向超平面投影，投影得到的向量就是最接近的。相当于平面外一点到平面的距离最短在垂线段的长度一样，只不过可能向量的维度是大于3的更高维度。一定能够找到后面的两个系数$b$和$a$。矩阵$M$的投影矩阵

$$
P=M(MM^T)^{-1}M^T
$$

投影在列空间上面的分量

$$
\boldsymbol{y_{projection}}=Py
$$

解方程组

$$
Mx=\boldsymbol{y_{projection}}
$$

得到的解就是最小二乘的解。其实也是投影向量在列空间的坐标值。

## 高维度下的推广
上面是以维度为2举例子，如果扩展到高维度($k$)，$y=b+a_1x_1 + a_2x_2 + \cdots + a_kx_k$，采样的点为$n$个。那么，$\boldsymbol e$的表达式为

$$
\boldsymbol{e}=\begin{bmatrix} e_1\\\  \vdots \\\ e_n    \end{bmatrix}=\begin{bmatrix} y_1-b-a_1x_{1,1}-a_2x_{2,1}- \cdots - a_kx_{k,1} 
  \\\ \vdots 
  \\ y_n-b-a_1x_{1,n}-a_2x_{2,n}- \cdots - a_kx_{k,n}   \end{bmatrix}
  =\begin{bmatrix} y_1\\\  \vdots \\ y_n   \end{bmatrix}-a_1\begin{bmatrix} x_{1,1}\\\  \vdots \\ x_{1,n}   \end{bmatrix}-
  a_2\begin{bmatrix} x_{2,1}\\\  \vdots \\ x_{2,n}   \end{bmatrix} - \cdots -
  a_k\begin{bmatrix} x_{k,1}\\\  \vdots \\ x_{k,n}   \end{bmatrix} - 
  b\begin{bmatrix} 1\\\  \vdots \\ 1\end{bmatrix}
$$

$$
=\begin{bmatrix} y_1\\\  \vdots \\ y_n   \end{bmatrix}-\begin{bmatrix} 1&x_{1,1}&\cdots&x_{k,1}\\\  \vdots&\vdots&\vdots &\vdots\\ 1 &x_{1,n}&\cdots&x_{k,n}  \end{bmatrix}\begin{bmatrix}b\\ a_1\\\ \vdots \\\ a_k\end{bmatrix}=\boldsymbol{y}-M\boldsymbol{x}
$$
可以看到，$M$矩阵的列扩充了$k-1$，没有本质的变化。

## 补充
在上面提到的最小化$\boldsymbol e$当中，其实就是求$M \boldsymbol x = \boldsymbol{y}$的最优解 $\boldsymbol{\hat x}$。这其实是线性回归的问题，在很多实际应用当中，都会介绍，目前，先不做展开，等遇到比较有趣的问题之后，我们再详细使用上面的工具去分析，去解决。

## 需要继续思考的内容
[Dot Product](https://www.youtube.com/watch?v=LyGKycYT2v0)
 这个视频讲解了向量点积的深层理解，说到了向量点积也可以理解为其中一个向量充当只有一行的变换矩阵，这个矩阵是用来投影的，并且还分析了矩阵里面的每一个元素其实就是空间基向量向这个向量的投影，还是这个向量向基向量的投影！因为投影具有对称的性质，这个视频很不错，值得一看。