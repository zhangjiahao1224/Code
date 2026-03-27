# PyTorch 教程（PyTorch Tutorials）

所有教程现在都以 Sphinx 风格的文档形式呈现：

## 文档地址

[`https://pytorch.org/tutorials`](https://pytorch.org/tutorials)

---

## 提问（Asking a question）

如果你对某个教程有问题，请在 `https://dev-discuss.pytorch.org/` 发帖，而不要在这个仓库里创建 Issue。你的问题在 dev-discuss 论坛上会得到更快的回复。

---

## 提交 Issue（Submitting an issue）

你可以提交以下类型的 Issue：

* **功能请求（Feature request）**：请求添加一个新的教程。请说明为什么需要这个教程，以及它是如何展示 PyTorch 价值的。
* **缺陷报告（Bug report）**：报告现有教程中的错误或过时信息。提交缺陷报告时，请运行：`python3 -m torch.utils.collect_env` 获取你的环境信息，并将输出结果附在缺陷报告中。

---

## 参与贡献（Contributing）

我们使用 sphinx-gallery 的[类 Notebook 风格示例](https://sphinx-gallery.github.io/stable/tutorials/index.html)来创建教程。语法非常简单，本质上你只需要写一个格式稍微规整一点的 Python 文件，它就会被展示为一个 HTML 页面。此外，还会自动生成一个 Jupyter Notebook，可在 Google Colab 中直接运行。

下面是创建新教程的步骤（详细说明见 [CONTRIBUTING.md](./CONTRIBUTING.md)）：

> 提交新教程前，请先阅读 [PyTorch 教程提交政策（PyTorch Tutorial Submission Policy）](./tutorial_submission_policy.md)。

1. 创建一个 Python 文件。如果你希望在构建文档时执行它，请将文件名后缀设为 `tutorial`，例如 `your_tutorial.py`。
2. 根据教程难度，将文件放入 `beginner_source`、`intermediate_source` 或 `advanced_source` 目录。如果是配方（Recipe），放入 `recipes_source`。对于展示不稳定原型特性的教程，放入 `prototype_source`。
3. 对于教程（原型特性除外），在 `toctree` 指令中包含该教程，并在 [index.rst](./index.rst) 中创建一个 `customcarditem`。
4. 对于教程（原型特性除外），在 [index.rst 文件](https://github.com/pytorch/tutorials/blob/main/index.rst)中，通过类似 `.. customcarditem:: beginner/your_tutorial.html` 的命令创建一个缩略卡片。对于配方（Recipes），在 [recipes_index.rst](https://github.com/pytorch/tutorials/blob/main/recipes_index.rst) 中创建缩略卡片。

如果你是从 Jupyter Notebook 开始的，可以使用[这个脚本](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe)将 Notebook 转换为 Python 文件。转换并添加到项目后，请确保章节标题等内容的顺序是合理的。

---

## 本地构建（Building locally）

完整构建所有教程的体量非常大，并且需要 GPU。如果你的机器没有 GPU 设备，你仍然可以在不实际下载数据或运行教程代码的情况下预览 HTML 构建结果：

1. 运行 `pip install -r requirements.txt` 安装所需依赖。

> 通常，你会在 `conda` 或 `virtualenv` 环境中运行。如果你想使用 `virtualenv`，在仓库根目录运行：`virtualenv venv`，然后执行 `source venv/bin/activate`。

- 如果你有一台带 GPU 的笔记本电脑，可以使用 `make docs` 进行构建。这会下载数据、执行所有教程，并将文档构建到 `docs/` 目录。对于带 GPU 的系统，这可能需要约 60–120 分钟。如果你的系统没有 GPU，请参考下一步。
- 你可以通过运行 `make html-noplot` 来跳过计算量大的图形生成步骤，只构建基础 HTML 文档到 `_build/html`，这样可以快速预览你的教程。

---

## 构建单个教程（Building a single tutorial）

你可以使用 `GALLERY_PATTERN` 环境变量来构建单个教程。例如，只运行 `neural_style_transfer_tutorial.py`，可以执行：

```bash
GALLERY_PATTERN="neural_style_transfer_tutorial.py" make html
```

或者：

```bash
GALLERY_PATTERN="neural_style_transfer_tutorial.py" sphinx-build . _build
```

`GALLERY_PATTERN` 变量支持正则表达式。

---

## 拼写检查（Spell Check）

你可以运行 `pyspelling` 来检查教程中的拼写错误：

- 仅检查 Python 文件：`pyspelling -n python`
- 仅检查 `.rst` 文件：`pyspelling -n reST`

目前 `.rst` 拼写检查仅覆盖 `beginner/` 目录，欢迎贡献以启用其他目录的拼写检查！

```bash
pyspelling              # 全量检查（约 3 分钟）
pyspelling -n python    # 仅检查 Python 文件
pyspelling -n reST      # 仅检查 reST 文件（目前只包含 beginner/ 目录）
```

---

## 关于为 PyTorch 文档和教程做贡献（About contributing to PyTorch Documentation and Tutorials）

* 你可以在 PyTorch 仓库的 [README.md](https://github.com/pytorch/pytorch/blob/master/README.md) 中找到关于为 PyTorch 文档做贡献的信息。
* 更多信息可以在 [PyTorch CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md) 中找到。

---

## 许可证（License）

PyTorch Tutorials 采用 BSD 许可证，具体内容见仓库中的 `LICENSE` 文件。
