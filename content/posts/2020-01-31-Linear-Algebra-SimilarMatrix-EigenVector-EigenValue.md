---
title:  相似矩阵和矩阵的特征值特征向量
date:   2020-01-31
categories:
  - Linear Algebra
author: Yang
tags:
  - Linear Algebra
  - Similar Matrix

math: true
---
## 相似矩阵

## 问题描述

为什么会产生**相似矩阵**这个感念？
场景描述：假定在一个线性空间当中，存在两组不同的基向量。在空间当中任何一个向量在不同的基下，投影的长度、方向都有所不同。当我们对这个向量施加变换的时候，也就是左乘一个变换矩阵，在不同的基下，这个左乘的矩阵形式上面是不同的，但是，它们有一个共同特点，就是相似，因为描述了同一个变换。
## 正向推导

按照我们刚才看到的这段话，我们可以用数学语言进行一次推导。

### 同一个向量不同基下的描述

假设线性空间内基由$n$个向量组成，我们不妨选取其中两组基：$\begin{bmatrix}\boldsymbol{\alpha_1} & \boldsymbol{\alpha_2} \cdots \boldsymbol{\alpha_n}  \end{bmatrix}$$，$$\begin{bmatrix}\boldsymbol{\beta_1} & \boldsymbol{\beta_2} \cdots \boldsymbol{\beta_n}  \end{bmatrix}$。

在此空间内的任意向量$\boldsymbol{b}$:

$$
\begin{equation}
\boldsymbol{b} = \begin{bmatrix}\boldsymbol{\alpha_1} & \boldsymbol{\alpha_2} \cdots \boldsymbol{\alpha_n}  \end{bmatrix} \boldsymbol{x_1}  \tag{1}
\end{equation}
$$

$$
\begin{equation}
\boldsymbol{b} = \begin{bmatrix}\boldsymbol{\beta_1} & \boldsymbol{\beta_2} \cdots \boldsymbol{\beta_n}  \end{bmatrix}\boldsymbol{x_2} \tag{2}
\end{equation}
$$

对于$\boldsymbol{b}$，有：

$$
\begin{equation}
[\boldsymbol{\alpha}]\boldsymbol{x_1}=[\boldsymbol{\beta}]\boldsymbol{x_2}
\label{eq3}\tag{3}
\end{equation}
$$

### 一次线性变换

对该向量做一次线性变换，分别得到在两组基下面的坐标向量：$\boldsymbol{y_1},\boldsymbol{y_2}$

那么，同一组基下，变换前后的两个向量具有如下的关系：

$$
\begin{equation}
T_1 \boldsymbol{x_1}=\boldsymbol{y_1}
\label{eq4}\tag{4}
\end{equation}
$$

$$
\begin{equation}
T_2 \boldsymbol{x_2}=\boldsymbol{y_2}
\label{eq5}\tag{5}
\end{equation}
$$


上面的关系先摆在这里，下面在推导相似矩阵的时候，会用到。

### 两组基的关系

建立两组基之间的联系：
既然是在同一空间内，$\boldsymbol{\beta_1}$可以通过$\boldsymbol{\alpha_1}, \cdots, \boldsymbol{\alpha_n}$的一个线性组合表达出来：

$$
\begin{equation}
\boldsymbol{\beta_1}=[\boldsymbol{\alpha}]\boldsymbol{p_1} 
\tag{6}
\end{equation}
$$

类似地，$[\boldsymbol{\beta}]$其他向量也可以由$[\boldsymbol{\alpha}]$线性组合得到。最终，得到下面的矩阵乘法形式：

$$
\begin{equation}
[\boldsymbol{\beta}]=[\boldsymbol{\alpha}] \begin{bmatrix} \boldsymbol{p_1} & \cdots & \boldsymbol{p_n}\end{bmatrix}=[\boldsymbol{\alpha}] P
\label{eq7}\tag{7}
\end{equation}
$$

### 两个变换矩阵的关系

将 ($\ref{eq7}$)代入($\ref{eq3}$)，得到：

$$
\begin{equation}
[\boldsymbol{\alpha}]\boldsymbol{x_1}=[\boldsymbol{\alpha}]P\boldsymbol{x_2}
\label{eq8}\tag{8}
\end{equation}
$$

整理 ($\ref{eq8}$)，得到：

$$
\begin{equation}
[\boldsymbol {\alpha}](\boldsymbol{x _1}-P\boldsymbol{x_2})=0
\label{eq9}\tag{9}
\end{equation}
$$

($\ref{eq9}$)中因为$[\boldsymbol{\alpha}]$各列线性无关，所以后面的向量只能是0向量。得到：


$$
\begin{equation}
\boldsymbol{x_1}=P\boldsymbol{x_2},
\boldsymbol{y_1}=P\boldsymbol{y_2}
\label{eq10}\tag{10}
\end{equation}
$$

联立($\ref{eq4}$) ($\ref{eq5}$)和($\ref{eq10}$)，消去$\boldsymbol{x_1}$、$\boldsymbol{y_1}$和$\boldsymbol{y_2}$

$$
\begin{equation}
( T_1P-PT_2 ) \boldsymbol{x_2}=0
\label{eq11}\tag{11}
\end{equation}
$$

此处请注意$\boldsymbol{x_2}$的任意性（在线性空间任意选择的这个向量）。所以得到前面的矩阵是一个零阵。

$$
\begin{equation}
T_1=PT_2P^{-1}
\label{eq12}\tag{12}
\end{equation}
$$

这里解释一下为什么$P$是可逆方阵。
$[\boldsymbol{\alpha}] \boldsymbol{x}=0$只有零解，同样的，$[\boldsymbol{\beta}] \boldsymbol{x}=0$也只有零解。将($\ref{eq7}$)代入前面的表达式替换$[\boldsymbol{\beta}]$，得到$[\boldsymbol{\alpha}]P\boldsymbol{x}=0$也只有零解。假设存在一个非零的$\boldsymbol{x_0}$使得$P\boldsymbol{x_0}=0$，那么$[\boldsymbol{\alpha}]P\boldsymbol{x}=0$有非零解，矛盾，所以$P\boldsymbol{x}=0$只有零解。从而，方阵$P$可逆。

### 小结

相似矩阵是一个“家庭”，其中
一个线性变换是一个函数。一旦确定了这个函数对定义域中的每一个对象的变换，也就确定了这个函数。所以，从这一点上看，变换和基也是“配套使用”。就像线性空间内的任意一个对象需要一组基和一个列向量。但是，线性空间内的每一个元素最小需要一组基来做**线性组合**。
通过刚才的步骤，相似矩阵是在不同的基下，描述同一个线性变换所对应的所有矩阵。这个矩阵的构造是我们从两个基开始的，每一个列向量中的元素是遍历一个基中的向量向另外一个基所有向量做投影之后得到的数。一步一步地完成了两个空间之间变换关系。请大家注意推导过程的($\ref{eq10}$)，同一个向量，在两个基中的表达不同，就需要$P$转换。适用于两组基下的所有向量。

## 矩阵的特征值和特征向量

定义：$n$维非零向量$\boldsymbol{v}$是$n \times n$方阵$A$的特征向量，当且仅当满足如下的等式

$$
\begin{equation}
A\boldsymbol{v}=\lambda \boldsymbol{v}
\end{equation}
\tag{13}
$$

关键点在于，某些特定的向量，经过矩阵$A$的变换，方向未发生变化，只是长度有了改变。假如存在$n$个这样的线性无关的向量，将它们按列排好，组成一个新的方阵，很容易得到

$$
\begin{equation}
AQ=Q\Lambda
\end{equation}
\label{eq14}\tag{14}
$$

其中:
$$\Lambda=\begin{bmatrix}\lambda_1 &0& \cdots &0\\\\ 0&\lambda_2& \cdots &0\\\\ \vdots&\vdots&\vdots&\vdots\\\\ 0&0&0&\lambda_n\end{bmatrix}$$

从而得到：$A=Q\Lambda Q^{-1}$。
$A$和$\Lambda$是一对相似矩阵。再回顾上面的小节，($\ref{eq7}$)和($\ref{eq12}$)，$[\boldsymbol{\beta}]$空间中的每一个基向量在$[\boldsymbol{\alpha}]$空间基向量的投影从上往下串成一个列向量，再把这些列向量按列排布。

划重点：
首先（推导到一半发现这里不太对，引起注意，这里容易出问题），需要明确，假设$\boldsymbol{\gamma}$在特征向量基下的坐标是$\boldsymbol{\gamma_{eig}}$，那么，有关系$I\boldsymbol{\gamma}=Q\boldsymbol{\gamma_{eig}}$（基变换的知识：不同基下相同对象的表达），所以得到$\boldsymbol{\gamma_{eig}}=Q^{-1}\boldsymbol{\gamma}$。
在由$n$个线性无关的单位正交向量组成的基下，某个变换$A$，在由$A$的特征向量这组基下，这个变换就是$\Lambda$。

更具体地说就是，任意给定一个向量$\boldsymbol{\gamma}$（单位正交基下），对$\boldsymbol{\gamma}$进行$A$的变换$IA\boldsymbol{\gamma}$，我们可以先投影到这个矩阵的特征向量上面得到坐标$\boldsymbol{\gamma_{eig}}=Q^{-1}\boldsymbol{\gamma}$，即找到在特征向量上面的系数，排列成一个列向量，到这里算是找到了在新基下的同一向量的表达，再左乘$\Lambda Q^{-1}\boldsymbol{\gamma}$，这一步是完成变换的过程，但是要说明基是什么，所以最后再在左边乘一个$Q$，即$Q\Lambda Q^{-1}\boldsymbol{\gamma}$。

在以特征向量为基底的变换，更为简单，只需要把待变换的向量做各个基的投影，得到投影向量，再利用特征值进行伸缩变换。但是在单位正交基底下，待变换的向量做完投影，没有办法直接进行伸缩变换，还是需要对投影向量做复杂的旋转、拉伸，这个过程不易理解，不可控制。
下面是几个问题：
1. 投影矩阵的特征值有哪些？有什么样的特性。

答：对于空间内的向量，经过投影矩阵的作用，可以分类为三类：第一类是在投影矩阵要投影的子空间，即上面所说的$A$的列空间，第二类是垂直于第一类的子空间，即$A$的左零空间，第三类是除了前面两类的其他。由于我们要找的是变换之后方向不变的向量，所以对于第一类，特征值是1，对于第二类，特征值是0。所以，一个投影矩阵只有0,1两个特征值。

2.两个相似矩阵的特征值有什么关系？

答：相似矩阵有相同的特征值。证明如下：矩阵$A$相似于矩阵$B$，那么，$B=SAS^{-1}$，$S$可逆。如果$\lambda$是$A$的任意一个特征值，有：$Ax=\lambda x$。我们对这个式子做一下简单的处理：$AS^{-1}Sx=\lambda x$，左右两边都左乘一个$S$，得到$(SAS^{-1})Sx=BSx=\lambda Sx$。所以，矩阵$B$也有同样的特征值$\lambda$。
