---
layout: post
title:  矩阵的奇异值分解
date:   2020-06-22
categories: Linear Algebra
author: Yang
tags: [Linear Algebra]
comments: true
toc: true
mathjax: true
---

> Besides having a rather simple geometric explanation, the singular value decomposition offers extremely effective techniques for putting linear algebraic ideas into practice.

# SVD分解与四个子空间
任意的线性变换都可以进行奇异值分解(SVD: Singular Value Decomposition)。之前的一篇博客讲的特征值和特征向量都是针对方阵而言的.这里任意形状的矩阵都能够进行SVD变换。给定一个$m \times n$的矩阵$A$，$rank(A)=r$，这里需要特别说明，我们可以在这个矩阵的行空间和列空间内找到$r$个线性无关的基，即使行、列空间的维度有不同。此处是核心：我们能够参考方阵，对它的特征值特征向量的推导过程得到启发，结合这里矩阵不一定是方阵（行、列维度不同的特点），能否在行子空间内找到一组相互正交的单位基向量，这组基向量通过$A$的变换在列空间内“生产”出一组也是相互正交的向量，并且这组向量正好是列空间的一组基。

$$
\begin{equation}
Av_1=\sigma_1u_1, \cdots, Av_r=\sigma_1u_r
\end{equation}
\label{eq1}\tag{1}
$$

($\ref{eq1}$) 中，$v_i(i=0,1,\cdots,r)$是行空间内的单位向量，且$v_i^T \dot v_j=0 (i \neq j)$，$u_i(i=0,1,\cdots,r)$是列空间内的单位向量（Orthonormal basis），且$u_i^T \dot u_j=0 (i \neq j)$，$\sigma_i(i=0,1,\cdots,r)$是列空间内单位向量的伸缩系数。
写成矩阵乘法的形式如下：

$$
\begin{equation}
A\left[ \boldsymbol{v_1}\ \boldsymbol{v_1}\cdots \boldsymbol{v_r} \right] =\left[ \boldsymbol{u_1}\ \boldsymbol{u_2}\cdots \boldsymbol{u_r} \right] \begin{bmatrix}\sigma_1&0&\cdots&0\\0&\sigma_2&\cdots&0\\\vdots&\vdots&\vdots&\vdots\\0&\cdots&\cdots&\sigma_r\end{bmatrix}
\end{equation}
\label{eq2}\tag{2}
$$

写成更简单的形式是：$AV=U\Lambda$。

![](/images/svd_4spaces.png)

## 奇异值分解的几何意义

接上面的矩阵定义$A$，我们看一下$R^n$空间的任意对象$\boldsymbol{x}$是如何变换的。

$$
\begin{equation}
\boldsymbol{x}=x_1\boldsymbol{v_1}+x_2\boldsymbol{v_2}+\cdots+x_n\boldsymbol{v_n}
\end{equation}
\tag{3}\label{eq3}
$$

($\ref{eq3}$)中，$x_i$是$\boldsymbol{x}$在行空间、零空间的坐标值，其中前$r$个向量是行空间的分量，后面的是零空间的分量。并且$\sum_{1}^{n}{x_i^2} =1$，可以把这个对象想象成在$R^n$内的单位球体。线性变换：

$$
\begin{equation}
A\boldsymbol{x} = x_1A\boldsymbol{v_1}+x_2A\boldsymbol{v_2}+\cdots+x_nA\boldsymbol{v_n} = \sigma_1x_1\boldsymbol{u_1}+\sigma_2x_2\boldsymbol{u_1}+\cdots+\sigma_rx_r\boldsymbol{u_r}
\end{equation}
\tag{4}\label{eq4}
$$

($\ref{eq4}$)少了后面的$(n-r)$个分量，是因为$A\boldsymbol{v_i}=0,(i=r+1,r+2,\cdots,n)$。
到这里，我们理解奇异值分解的几何意义就是：矩阵的奇异值分解，是在行空间内找到一组相互垂直的单位向量，这组向量经过矩阵的变换在列空间得到的向量也是相互垂直，观察($\ref{eq4}$)，旧坐标值和奇异值相乘的意义就是在新的基方向上，进行奇异值大小的拉伸操作。更加形象的理解就是，将单位球面变换为超椭球面，这个球面具有r$个半轴，每个半轴对应的奇异值是该半轴伸缩的系数。



最近在知乎上又看到了[硬核机器学习文章分享](https://www.zhihu.com/people/linghan-cheung)的回答，在他的回答中，两组正交基分别就是矩阵行空间、列空间的标准正交基。如果不好记忆的话，我们可以对比矩阵和向量的乘法，$\boldsymbol{x}$向量的维度和矩阵的行向量是一致的，所以 $\boldsymbol{v_i}$ 向量是用来分解给定的向量的，这样我们就很容易得到酉矩阵$U$的列向量是原矩阵列向量的一组标准正交基构成。

![](/images/svd_process.png)

### 总结一下

步骤如下：

- 给出任意向量$\boldsymbol{x}$


- 矩阵对该向量的变换分解为如下

- 投影$\boldsymbol{x}$到$\boldsymbol{v_i}$，得到投影矢量为$\epsilon_1,\cdots,\epsilon_r$


- 使用$\boldsymbol{V}$旋转行基$\boldsymbol{v_i}$到单位标准正交基$\boldsymbol{I}$


- 使用奇异值拉伸$\boldsymbol{I}$的各个分量


- 使用$\boldsymbol{U}$旋转拉伸后的基向量，得到最终基，变换在该基下的分矢量就是$\epsilon_1,\cdots,\epsilon_r$


## 和特征值分解的关系
刚才只讨论了矩阵行空间和列空间的向量，那么，剩下的零空间和左零空间的向量难道就放弃了么？并不是这样的，出于考虑问题的完备性，我们写出如下的表达式

$$
\begin{equation}
A=[\boldsymbol{u_1}\ \boldsymbol{u_2}\cdots \boldsymbol{u_r}|\boldsymbol{u_{r+1}}\ \boldsymbol{u_{r+2}}\cdots \boldsymbol{u_m}]
\begin{bmatrix}
    \sigma_1&0&\cdots&0
    \\0&\sigma_2&\cdots&0
    \\\vdots&\vdots&\vdots&\vdots
    \\0&\cdots&\cdots&\sigma_r
    \\0&\cdots&\cdots&0
    \\\vdots&\vdots&\vdots&\vdots
    \\0&\cdots&\cdots&0
\end{bmatrix}
\begin{bmatrix}
    \boldsymbol{v_1^T}
    \\\boldsymbol{v_2^T}
    \\\vdots
    \\\boldsymbol{v_r^T}
    \\\boldsymbol{v_{r+1}^T}
    \\\vdots
    \\\boldsymbol{v_n^T}
\end{bmatrix}
\label{eq:5}
\end{equation}
$$

SVD更广泛的意义是将$R^n$（行空间+零空间）内的一组标准正交基和$R^m$（列空间+左零空间）的标准正交基拉进来，形式上完备了，具体矩阵所具有的特性取决于中间对角矩阵中非零奇异值的个数，理想的情况是，当矩阵是一个$n\times n$方阵时，并且具有$n$个特征向量，那么行空间和列空间的维度是一样的，通过SVD分解矩阵得到的$U$中，各列就是该矩阵的特征向量。从这点上，我们可以得到，方阵的特征向量特征值是矩阵SVD分解的特殊情况。

## 参考材料

[We Recommend a Singular Value Decomposition](http://www.ams.org/publicoutreach/feature-column/fcarc-svd)	"奇异值分解的直观理解"

[机器学习中的数学(5)：强大的矩阵奇异值分解(SVD)及其应用](https://mp.weixin.qq.com/s?__biz=MzA5ODUxOTA5Mg==&mid=211203099&idx=1&sn=c741c2c535a5042a03cd1b3a83a1b654&scene=20#wechat_redirect%20%E5%BC%BA%E5%A4%A7%E7%9A%84%E7%9F%A9%E9%98%B5%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3%28SVD%29%E5%8F%8A%E5%85%B6%E5%BA%94%E7%94%A8)

[线性变换的矩阵为什么要强调在这组基下?](https://www.zhihu.com/question/22218306/answer/88697757)

[The SVD of a Matrix](http://www-users.math.umn.edu/~lerman/math5467/svd.pdf)


