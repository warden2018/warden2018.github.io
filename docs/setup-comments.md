# Hugo Blog 留言系统设置指南

本文档介绍了如何为 Hugo PaperMod 博客设置留言系统。

## 快速开始

### 1. 选择留言系统

#### Utterances（推荐）
- 基于 GitHub Issues
- 无需服务器，免费
- 支持 GitHub 身份认证
- 自动适配深色/浅色主题

#### Giscus
- 基于 GitHub Discussions
- 支持更多功能（回复、点赞等）
- 需要 GitHub Discussions 权限
- 适合更活跃的社区

### 2. 配置步骤

#### 对于 Utterances：

1. 在 GitHub 仓库安装 [utterances](https://github.com/utterances/utterances) 应用
2. 确保 GitHub 账户有访问仓库的权限
3. 在 hugo.toml 中已配置好参数（默认已启用）

#### 对于 Giscus：

1. 在 GitHub 仓库启用 Discussions
2. 访问 [Giscus](https://giscus.app/) 生成配置
3. 将配置更新到 hugo.toml

### 3. 在文章中启用评论

默认情况下，所有文章都会显示评论。如果需要在某些文章中禁用评论，在文章的 frontmatter 中添加：

```yaml
---
title: "文章标题"
comments: false
---
```

## 自定义样式

评论模板已经包含了基本的样式修复：

- 适配 PaperMod 主题宽度
- 修复主题切换时的样式问题
- 优化移动端显示

## 注意事项

1. **隐私考虑**：这些评论系统会收集访客的 GitHub 信息
2. **依赖 GitHub**：如果 GitHub 不可用，评论会失效
3. **加载速度**：评论系统需要从 GitHub 加载脚本，可能会有轻微延迟

## 高级配置

如需更高级的配置，可以编辑 `themes/PaperMod/layouts/partials/comments.html` 文件。