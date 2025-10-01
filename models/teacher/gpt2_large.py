import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPT2TeacherLarge:
    def __init__(self, model_name="gpt2-medium", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Teacher] Loading {model_name} on {self.device}...")

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def get_logits(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.9, top_p=0.95, top_k=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    teacher = GPT2TeacherLarge()
    print("GPT-2 Large interactive mode. Type /exit to quit.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            break
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if not prompt:
            continue
        text = teacher.generate(prompt)
        print(text)
