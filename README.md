# ChatLaw-法律大模型

<div align="center">
  <a href="https://github.com/PKU-YuanGroup/ChatLaw">
  <img src="https://raw.githubusercontent.com/PKU-YuanGroup/ChatLaw/main/images/logo.png" width="50%">
  </a>

  <p align="center">
    <h3>但愿世间不纷争，何惜法典卷生尘</h3>
      <a href="https://github.com/PKU-YuanGroup/ChatLaw/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/PKU-YuanGroup/ChatLaw" />
      </a>
      <a href="https://github.com/PKU-YuanGroup/ChatLaw/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/PKU-YuanGroup/ChatLaw?color=0088ff" />
      </a>
      <a href="https://github.com/PKU-YuanGroup/ChatLaw/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/PKU-YuanGroup/ChatLaw?color=0088ff" />
      </a>
      <a href="https://github.com/PKU-YuanGroup/ChatLaw/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/PKU-YuanGroup/ChatLaw?color=ccf" />
      </a>
      <br/>
      <em>易用 / 简单 / 快速 </em>
      <br/>
      <a href="https://chatlaw.cloud/"><strong>官网</strong></a>
        ·
      <a href="https://chatlaw.cloud/lawchat/"><strong>在线体验</strong></a>
    </p>
  </p>
</div>


# ChatLaw系列模型

- HuggingFace: [ChatLaw-13B](https://huggingface.co/JessyTsu1/ChatLaw-13B)
- HuggingFace: [ChatLaw-33B](https://huggingface.co/JessyTsu1/ChatLaw-33B)
- HuggingFace: [ChatLaw-Text2Vec](https://huggingface.co/chestnutlzj/ChatLaw-Text2Vec)

## 简介 Brief Introduction

ChatLaw法律大模型目前开源的仅供学术参考的版本为姜子牙-13B版本、Anima-33B版本，我们使用大量法律新闻、法律论坛、法条、司法解释、法律咨询、法考题、判决文书等原始文本来构造对话数据。

基于姜子牙-13B的模型是第一版模型，得益于姜子牙的优秀中文能力和我们对数据清洗、数据增强过程的严格要求，我们在逻辑简单的法律任务上表现优异，但涉及到复杂逻辑的法律推理任务时往往表现不佳。

随后基于[Anima-33B](https://github.com/lyogavin/Anima)，我们增加了训练数据，做成了ChatLaw-33B，发现逻辑推理能力大幅提升，由此可见，大参数的中文LLM是至关重要的。

基于可商用的模型训练而成的版本会作为我们产品[ChatLaw](http://www.chatlaw.cloud/)内部接入的版本，对外不开源。

## 效果 Results

![](https://raw.githubusercontent.com/PKU-YuanGroup/ChatLaw/main/images/demo_3.jpg)
![](https://raw.githubusercontent.com/PKU-YuanGroup/ChatLaw/main/images/demo_1.jpg)
![](https://raw.githubusercontent.com/PKU-YuanGroup/ChatLaw/main/images/demo_2.jpg)


##  使用 Usage

由于LLaMA权重的许可限制，该模型不能用于商业用途，请严格遵守LLaMA的使用政策。考虑到LLaMA权重的许可证限制，我们无法直接发布完整的模型权重。、



欢迎引用我们的[网站](https://github.com/PKU-YuanGroup/ChatLaw):

```
@misc{ChatLaw,
  author={Jiaxi Cui and Zongjian Li and Yang Yan and Bohua Chen and Li Yuan},
  title={ChatLaw},
  year={2023},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/PKU-YuanGroup/ChatLaw}},
}
```



