[中文](README-zh.md)   |    [English](README.md)

demo data:ml-1m 

finetuninig:基于unsloth 使用llama3-8B-bnb模型搭建的微调框架，排序，在候选物品中生成式推荐物品文本

模型的输出
<Lora LLM output> :
Based on the provided user profile and candidate items, the top 5 items that the user may be interested in are:

1. 1094: Drama | Romance | War
2. 34: Children's | Comedy | Drama
3. 1179: Crime | Drama | Film-Noir
4. 3125: Drama
5. 503: Drama<|end_of_text|>


真实答案
<actual output>:

"interested items: 
1094:Drama|Romance|War;
34:Children's|Comedy|Drama;
1179:Crime|Drama|Film-Noir;
3125:Drama;
3298:Drama"