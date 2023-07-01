import fire
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import gradio as gr
import torch
import re

def make_prompt(
    references: str = "",
    consult: str = ""
):
    prompt = "" if references == "" else f"References:\n{references}\n"
    prompt += f"Consult:\n{consult}\nResponse:\n"
    return prompt

def main(
    model: str = "JessyTsu1/ChatLaw-13B",
):

    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        
    model.eval()
    
    def evaluate(
        references,
        consult,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = make_prompt(references, consult)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.2,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        if search_result := re.search("Response\s*:\s*([\s\S]+?)</s>", output):
            return search_result.group(1)
        return "Error! Maybe response is over length."

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=4,
                label="References",
                placeholder="输入你的参考资料",
            ),
            gr.components.Textbox(
                lines=2,
                label="Consult",
                placeholder="输入你的咨询内容，在问题前加上“详细分析：”会有更好的效果。",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.7, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=1, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=1024, step=1, value=1024, label="Max tokens"
            ),
        ],
        outputs = [
            gr.inputs.Textbox(
                lines=8,
                label="Response",
            )
        ],
        title="ChatLaw Academic Demo",
        description="",
    ).queue().launch(server_name="0.0.0.0",server_port=1234)


if __name__ == "__main__":
    fire.Fire(main)
