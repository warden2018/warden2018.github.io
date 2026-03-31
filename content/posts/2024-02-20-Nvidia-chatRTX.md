---
title: Nvidia Chat With RTX
date:   2024-02-20
categories:
  - AI
author: Yang
tags:
  - GPT offline
math: true
---

## 安装部署

[软件下载地址](https://us.download.nvidia.com/RTX/NVIDIA_ChatWithRTX_Demo.zip)

![安装界面](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_install.png)

一路点击下一步安装即可。

![安装界面2](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_install2.png)

安装过程需要耐心等待，我可能等了差不多两个小时，最后安装成功。

安装完成之后，在桌面会出现`NVIDIA Chat with RTX`快捷方式，就是一个启动脚本。
如果是默认路径安装的话，需要修改`path`：


`C:\Users\Yang\AppData\Local\NVIDIA\ChatWithRTX\RAG\trt-llm-rag-windows-main\config\preferences.json`

```
{
  "dataset": {
    "path": "C://Users//Yang//Desktop//rtx",
    "isRelative": false
  }
}
```

![加参数](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_share_params.png)


## 运行

![加载模型和参数](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_install_running.png)

![根据路径下文件内容生成embeddings](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_install_running_finish.png)

这里有一个问题就是：如果我的数据库当中增加或者减少文件，需要重新生成embeddings,这个过程很慢。

![GPU运行情况](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_GPU.png)

## 问问题

由于我第一次指定的`path`下文件数量很大，导致无法全部加载成功，我减少文件数量，单独加载一个关于bike fitting的书。

![测试](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_test.png)


![基础比例对判断的影响](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_cabs_accidents1.png)

![使用贝叶斯定理计算概率1](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_cabs_accidents2.png)

![使用贝叶斯定理计算概率2](https://images-1302340771.cos.ap-beijing.myqcloud.com/images/Chat_with_RTX_cabs_accidents3.png)

## 感受
同样的出租车问题，我几乎使用相同的单词来询问，给出的答案不统一。稳定性不够好。其他的测试后面再补充。
