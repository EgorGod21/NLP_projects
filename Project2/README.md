# Project_2

Link to article ["Simplifying Transformer Blocks"](https://arxiv.org/abs/2311.01906)

Models shown: 
- Pre-LN
- Parallel
- SAS (Simplified Attention Sub-block)
- SAS-P (Parallel Simplified Attention Sub-block)

![models.png](images%2Fmodels.png)

**For train.py:**
```
input: python train.py pre_ln
output: step 0: train loss 4.7158, val loss 4.7169
step 99: train loss 2.7487, val loss 2.7542
```
**For generate.py:**
```
input: python generate.py pre_ln model_pre_ln.pth
output: Такси всу, Алксос, наменна им. Обли питроков дохоро,

Слетной хотьм увида ответленами, созвраву Атрнивить

85 Приаелые стен обалися и Одно, раз сладый;
```

**The data presented was obtained during a 5k step train**

![parameters.png](images%2Fparameters.png)
![time.png](images%2Ftime.png)
![perplexity.png](images%2Fperplexity.png)
![loss_val_comparison.png](images%2Floss_val_comparison.png)