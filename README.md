---
license: other
language:
- en
pipeline_tag: text-generation
---
## Model Summary

The language model phi-1.5 is a Transformer with 1.3 billion parameters. It was trained using the same data sources as [phi-1](https://huggingface.co/microsoft/phi-1), augmented with a new data source that consists of various NLP synthetic texts. When assessed against benchmarks testing common sense, language understanding, and logical reasoning, phi-1.5 demonstrates a nearly state-of-the-art performance among models with less than 10 billion parameters.

We did not fine-tune phi-1.5 either for instruction following or through reinforcement learning from human feedback. The intention behind crafting this open-source model is to provide the research community with a non-restricted small model to explore vital safety challenges, such as reducing toxicity, understanding societal biases, enhancing controllability, and more.

For a safer model release, we exclude generic web-crawl data sources such as common-crawl from the training. This strategy prevents direct exposure to potentially harmful online content, enhancing the model's safety without RLHF. However, the model is still vulnerable to generating harmful content. We hope the model can help the research community to further study the safety of language models.

## Intended Uses
Given the nature of the training data, phi-1.5 is best suited for prompts using the QA format, the chat format, and the code format:

#### QA format:

```markdown
Write an analogy between a mind and a lighthouse.

Answer: A mind is like a lighthouse, guiding us through the darkness of ignorance and fear.
```
where the model generates the text after "Answer:".

#### Chat format:

```markdown
Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?

Bob: Have you tried using a timer? It can help you stay on track and avoid distractions.
```
where the model generates the text after "Bob:".

#### Code format:
~~~python
```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """
   primes = []
   for num in range(2, n+1):
       is_prime = True
       for i in range(2, int(num**0.5)+1):
           if num % i == 0:
               is_prime = False
               break
       if is_prime:
           primes.append(num)
   print(primes)

print_prime(20)
```
~~~
where the model generates the text after the comments. (Note: This is a legitimate and correct use of the else statement in Python loops.)

**Notes**
* phi-1.5 is intended for research purposes. The model-generated text/code should be treated as a starting point rather than a definitive solution for potential use cases. Users should be cautious when employing these models in their applications.
* Direct adoption for production tasks is out of the scope of this research project. As a result, phi-1.5 has not been tested to ensure that it performs adequately for any production-level application. Please refer to the limitation sections of this document for more details. 

## Limitations of phi-1.5

* Generate Inaccurate Code and Facts: The model often produces incorrect code snippets and statements. Users should treat these outputs as suggestions or starting points, not as definitive or accurate solutions.
* Limited Scope for code: Almost all codes in our dataset are Python and use only the packages "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.
* Unreliable Responses to Instruction: The model has not undergone instruction fine-tuning. As a result, it may struggle or fail to adhere to intricate or nuanced instructions provided by users.
* Language Limitations: The model is primarily designed to understand standard English.  Informal English, slang, or any other language outside of English might pose challenges to its comprehension, leading to potential misinterpretations or errors in response.
* Potential Societal Biases: Regardless of the safe data used for its training, the model is not entirely free from societal biases. There's a possibility it may generate content that mirrors these societal biases, particularly if prompted or instructed to do so. We urge users to be aware of this and to exercise caution and critical thinking when interpreting model outputs.
* Toxicity: Despite that the model is trained with carefully selected data, the model can still produce harmful content if explicitly prompted or instructed to do so. We chose to release the model for research purposes only -- We hope to help the open-source community develop the most effective ways to reduce the toxicity of a model directly after pretraining.

## Training

### Model (phi-1.5)
* Architecture: a Transformer-based model with next-word prediction objective
* Dataset size: 30B tokens
* Training tokens: 150B tokens
* Precision: fp16
* GPUs: 32xA100-40G
* Training time: 8 days

### Software
* [PyTorch](https://github.com/pytorch/pytorch)
* [DeepSpeed](https://github.com/microsoft/DeepSpeed)
* [flash-attention](https://github.com/HazyResearch/flash-attention)

### License
The model is licensed under the [Research License](https://huggingface.co/microsoft/phi-1_5/resolve/main/Research%20License.docx).

### Sample Code
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device('cuda')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
inputs = tokenizer('''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

eos_token_id = tokenizer.encode("``<|endoftext|>") # generation ends at `` or <|endoftext|>
outputs = model.generate(**inputs, max_length=500)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

### Citation
```bib
@article{textbooks2,
  title={Textbooks Are All You Need II: \textbf{phi-1.5} technical report},
  author={Li, Yuanzhi and Bubeck, S{\'e}bastien and Eldan, Ronen and Del Giorno, Allie and Gunasekar, Suriya and Lee, Yin Tat},
  year={2023}
}
```