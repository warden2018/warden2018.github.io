# Utterances 设置详细指南

Utterances 是一个基于 GitHub Issues 的轻量级评论系统，非常适合 Hugo 静态博客。

## 1. 安装 Utterances 应用

1. 访问 [utterances GitHub 应用页面](https://github.com/apps/utterances)
2. 点击 "Configure" 选择你的 GitHub 仓库
3. 确认安装权限

## 2. 配置参数说明

在 `hugo.toml` 中的 Utterances 配置：

```toml
[params.comments.utterances]
  enable = true
  repo = "你的GitHub用户名/你的仓库名"  # 例如：warden2018/warden2018.github.io
  issueTerm = "pathname"  # 使用页面路径作为 issue 标题
  theme = "github-light"  # 主题选项
```

### 参数详解：

- **repo**: 你的 GitHub 仓库地址，格式为 `owner/repo`
- **issueTerm**:
  - `pathname`：根据页面路径创建 issue（推荐）
  - `title`：根据文章标题创建 issue
  - `url`：根据页面 URL 创建 issue
  - `specific-post-title-identifier`：特定文章的标识
- **theme**:
  - `github-light`: 浅色主题
  - `github-dark`: 深色主题
  - `preferred-color-scheme`: 自动跟随系统主题（推荐）

## 3. 工作原理

Utterances 会在你的 GitHub 仓库中为每篇文章创建一个 issue：

- 当有人发表评论时，会作为 comment 添加到对应的 issue 中
- 评论支持 Markdown 格式
- 可以使用 GitHub reactions 表达点赞

## 4. 自定义显示

如果你想自定义评论显示，可以修改 CSS：

```css
/* 在 custom.css 中添加 */
.utterances {
    margin-top: 2rem;
    border-radius: 0.5rem;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.utterances h4 {
    margin-bottom: 1rem;
    color: var(--content);
}
```

## 5. 隐私设置

注意：访客必须登录 GitHub 才能发表评论。评论数据存储在你的 GitHub Issues 中。

## 6. 常见问题

### Q: 如何删除某个评论？
A: 在对应的 GitHub Issue 中删除对应的 comment。

### Q: 如何修改评论框的样式？
A: 修改 `comments.html` 文件中的 CSS 部分或在 `assets/scss/custom.scss` 中添加样式。

### Q: 如何禁用某些文章的评论？
A: 在文章的 frontmatter 中添加 `comments: false`。