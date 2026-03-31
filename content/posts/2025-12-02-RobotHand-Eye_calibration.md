---
title: 手眼标定
date: 2025-12-02
categories:
  - State Estimation
author: Yang
tags:
  - 标定
math: true
---

本文参考了[OPENCV](https://www.torsteinmyhre.name/snippets/robcam_calibration.html)

## 问题描述
在机器人系统中，经常遇到需要确定相机（眼睛）与机器人末端（TCP）之间的安装关系。如果相机不在机械臂末端，往往末端会安装一个相机能够识别的标记物，相机系统给出该标记物在相机空间的三维坐标和姿态。无论上述哪种安装类型，都需要确定一个方程的解：$AX=XB$，$A,B$是已知的齐次矩阵，$X$是未知的齐次矩阵。

![](https:/https://images-1302340771.cos.ap-beijing.myqcloud.com/images-1302340771.cos.ap-beijing.myqcloud.com/extrinsic-camera-calibration-stationary-camera.png)

利用李代数和最小二乘解决$AX=XB$的问题。

## 平移旋转分开求解

## 从齐次等式提取旋转部分
$$
\begin{bmatrix}
R_A & b_A \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
R_X & b_X \\
0 & 1
\end{bmatrix}=
\begin{bmatrix}
R_X & b_X \\
0 & 1
\end{bmatrix}\begin{bmatrix}
R_B & b_B \\
0 & 1
\end{bmatrix}
$$

$$
R_AR_X=R_XR_B
$$

## 李代数和李群之间的相互映射
在SO(3)上，旋转矩阵$R$的李代数$\phi$ $\boldsymbol u\in so(3)$是一个三维向量($\boldsymbol u$是一个单位向量，$\phi\in \mathbb R$)。$\boldsymbol u^{\wedge}$是一个反对称矩阵。

### so(3)李代数到SO(3)的指数映射--罗德里格斯公式

$$
R=e^{\phi \boldsymbol u^{\wedge}}=\cos\phi I + (1-\cos \phi) \boldsymbol u\boldsymbol u^T + \sin \phi \boldsymbol u^{\wedge}
$$

### SO(3)到so(3)的对数映射

$$
log(R)=\boldsymbol \phi^{\wedge}=\frac{\phi}{2\sin\phi}(R-R^T)
$$

其中，

$$
\phi = \arccos\left(\frac{trace(R)-1}{2}\right)
$$

### 李代数$\boldsymbol u=[u_1 \ u_2 \ u_3]^T$对应的反对称矩阵的定义和性质
$$
\boldsymbol u^{\wedge}=
\begin{bmatrix}
0 & -u_3 & u_2 \\
u_3 & 0 & -u_1 \\
-u_2 & u_1 & 0
\end{bmatrix}
$$

***并且，该反对称矩阵有性质：$R=e^{\phi\boldsymbol u^{\wedge}}$(体现了$\phi \boldsymbol u^{\wedge}$是旋转对时间导数的性质，$\phi$越大，意味着绕着转轴$\boldsymbol u$转动越快速)。***

## 李群相等转换到李代数相等

问题转化：找到一个$X$，使得李群中的旋转$R_A$和$R_XR_BR_X^T$相等（这里是理想情况下，测量不带噪声）。
如果测量值$R_{A_i}$，$R_{B_i}$不带噪声，给定任意一组测量值$R_A$和$R_B$。看看李代数so(3)中的两个向量怎么和上述的旋转相等建立关系。

$$
\begin{aligned}
& R_AR_X=R_XR_B \\
& R_A=R_XR_BR_X^T \\
& log(R_A)=log(R_XR_BR_X^T) \leftarrow \boldsymbol \phi_A^\wedge=log(R_A),\boldsymbol \phi_B^\wedge=log(R_B)\\
& \boldsymbol \phi_A^\wedge=R_X\boldsymbol \phi_B^\wedge R_X^T \leftarrow (R_X \boldsymbol \phi_B)^\wedge=R_X\boldsymbol \phi_B^\wedge R_X^T\\
& \boldsymbol \phi_A=R_X\boldsymbol \phi_B
\end{aligned}
$$

因为绕着一个轴旋转$\theta$和$\theta + 2\pi$角度效果是一样的，上面的推导有一个条件，就是约束$||\boldsymbol \phi_A|| \in (-\pi,\pi]$,$||\boldsymbol \phi_B|| \in (-\pi,\pi]$。

## 求解过程
## 测量带噪声
在给定数据带有噪声的情况下，我们把每一组转化之间的误差进行平方和最小化：
$$
\arg_{R_X \in SO(3)}min\sum_i ||\boldsymbol \phi_{A_i} - R_X \boldsymbol \phi_{B_i}||^2
$$

根据[Nadas](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://dominoweb.draco.res.ibm.com/reports/RC6945.pdf)：

$$
\eta=\sum_{i=1}^{N}||\Theta\boldsymbol x_i+\boldsymbol b -\boldsymbol y_i||
$$

当且仅当$\Theta = (M^TM)^{-1/2}M^T$使得$\eta$最小，其中，$M=\sum_{i=0}^{i=N}{x_i y_i^T}$

## 数据预处理
采集了$\{R_{A_0},R_{A_1},\cdots,R_{A_n}\}$和$\{R_{B_0},R_{B_1},\cdots,R_{B_n}\}$数据。先根据上面的方法，求出$\alpha_i=log(R_{A_i})$和$\beta_i=log(R_{B_i})$。
将$\alpha$和$\beta$组成矩阵$M$:

$$
M=\sum_{i=0}^{i=N}{\beta_i \alpha_i^T}
$$

## 求解旋转矩阵
先求旋转参数$R^\star_X$

$$
R^\star_X = (M^TM)^{-1/2}M^T
$$

## 求解平移向量
还有平移的参数需要估计$\boldsymbol b_X$。

$$
\begin{aligned}
& R_A\boldsymbol b_X+\boldsymbol b_A=R^\star_X\boldsymbol b_B+\boldsymbol b_X \\
& \rightarrow
\eta_2=\sum_{i=1}^{N}||(R_{A_i}-I)\boldsymbol b_X-R^\star_X\boldsymbol b_{B_i} +\boldsymbol b_{A_i}||
\end{aligned}
$$
上面的最小化，是典型的最小二乘问题:[投影矩阵和最小二乘](https://warden2018.github.io/2020/01/30/2020-01-30-Linear-Algebra-ProjectionMatrix-Least-Squares/)

$$
\begin{aligned}
& M_i=R_{A_i}-I,\boldsymbol y_i=R^\star_X\boldsymbol b_{B_i} -\boldsymbol b_{A_i} \\
& C=
\begin{bmatrix}
M_1 \\
M_2 \\
\vdots \\
M_N
\end{bmatrix} \\
& Y=\begin{bmatrix}
\boldsymbol y_1 \\
\boldsymbol y_2 \\
\vdots \\
\boldsymbol y_N
\end{bmatrix} \\
& \boldsymbol b_X^\ast= \arg_{\boldsymbol b_X}min(C\boldsymbol b_X-Y)=C^{-1}C(CC^T)^{-1}C^T=(CC^T)^{-1}C^T
\end{aligned}
$$

到此，旋转和平移全部求出。

## 代码
```C++
void park_martin(std::vector<Eigen::Matrix4d> A, std::vector<Eigen::Matrix4d> B, Eigen::Matrix4d& result)
{
    int dataLength = A.size();
    if(dataLength != B.size())
    {
        return;
    }

    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();

    for (int i = 0; i < dataLength; i++)
    {
        Eigen::Matrix3d Ra = A[i].toMatrix().block<3, 3>(0, 0);
        Eigen::Matrix3d Rb = B[i].toMatrix().block<3, 3>(0, 0);
        M += log(Rb) * log(Ra).transpose();
    }

    //best rotation
    Eigen::Matrix3d theta_X = invsqrt(M.transpose() * M) * M.transpose();

    // best translation
    Eigen::MatrixXd C(3 * dataLength, 3);
    Eigen::MatrixXd y(3 * dataLength, 1);

    for (int i = 0; i < dataLength; i++)
    {
        Eigen::Matrix3d Ra = A[i].toMatrix().block<3, 3>(0, 0);
        Eigen::Vector3d b_a = A[i].toMatrix().block<3, 1>(0, 3);

        Eigen::Matrix3d Rb = B[i].toMatrix().block<3, 3>(0, 0);
        Eigen::Vector3d b_b = B[i].toMatrix().block<3, 1>(0, 3);
        C.block<3, 3>(i * 3, 0) = Ra - Eigen::Matrix3d::Identity();
        y.block<3, 1>(i * 3, 0) = theta_X * b_b - b_a;
    }
    Eigen::Vector3d b_x = (C.transpose() * C).inverse() * C.transpose() * y;

    Eigen::Matrix4d result_M = Eigen::Matrix4d::Identity();
    result_M.block<3, 3>(0, 0) = theta_X;
    result_M.block<3, 1>(0, 3) = b_x;
    result = result_M;
}

Eigen::Vector3d DrsToolCalibration::log(Eigen::Matrix3d rotM)
{
    double theta = std::acos((rotM(0, 0) + rotM(1, 1) + rotM(2, 2) - 1.0) / 2.0);
    return Eigen::Vector3d(rotM(2, 1) - rotM(1, 2), rotM(0, 2) - rotM(2, 0), rotM(1, 0) - rotM(0, 1))* theta / (2 * std::sin(theta));
}

Eigen::Matrix3d DrsToolCalibration::invsqrt(Eigen::Matrix3d rotM)
{
    if (rotM.rows() != rotM.cols()) {
        return Eigen::MatrixXd();
    }
   
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(rotM, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd s = svd.singularValues();
    for (int i = 0; i < s.size(); ++i) {
        if (s(i) <= 0) {
            break;
        }
    }
    Eigen::VectorXd inv_sqrt_s = s.array().inverse().sqrt();
    return svd.matrixU() * inv_sqrt_s.asDiagonal() * svd.matrixV().transpose();
}
```
如何使用：

```C++
    Eigen::Matrix4d park_martin_result;
    std::vector<Eigen::Matrix4d> tcp_list;
    std::vector<Eigen::Matrix4d> rh_list;

    // here fill tcp_list and rh_list with data

    park_martin(rh_list, tcp_list, park_martin_result);

    // park_martin_result is the hand-eye calibration result
```

