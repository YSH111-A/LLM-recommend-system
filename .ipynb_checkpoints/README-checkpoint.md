[中文](README-zh.md)  ｜  [English](README.md)

Demo data: ml-1m

finetuning: A fine-tuning framework based on the llama3-8B-bnb model built by Unsloth, which ranks and generates text recommendations among candidate items.

<Lora LLM output> :
Based on the provided user profile and candidate items, the top 5 items that the user may be interested in are:

1. 1094: Drama | Romance | War
2. 34: Children's | Comedy | Drama
3. 1179: Crime | Drama | Film-Noir
4. 3125: Drama
5. 503: Drama<|end_of_text|>

<actual output>:

"interested items: 
1094:Drama|Romance|War;
34:Children's|Comedy|Drama;
1179:Crime|Drama|Film-Noir;
3125:Drama;
3298:Drama"
