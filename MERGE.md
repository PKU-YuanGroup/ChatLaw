# 权重合并说明

由于LLaMa权重许可证限制，我们不可以把完整模型直接发布，因此需要手动合并。合并流程如下：

Step1：获取原始LLaMa模型权重hf版本。（可以到huggingface上搜索`llama-7b`下载）

Step2：合并[Ziya](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)权重

Step3：合并ChatLaw权重（我们开源的为LoRA权重，可以参考其他repo中的LoRA权重合并方式，例如[Chinese-LLaMa-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)）

# 说明

我们没有开源Keyword LLM以及法律数据库，因此您部署的模型只有简单的对话功能，正确性较低，不能用于真实法律场景。

我们开源此模型的目的是为了便于各位在我们的学术Demo模型的基础上注入自己的优质数据，可以用更短的微调时间来微调出自己的专业领域模型。