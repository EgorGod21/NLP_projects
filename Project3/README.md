**Project_3 – Binary russian review classification**

[Dataset](https://huggingface.co/datasets/merkalo-ziri/vsosh2022)

[Model](https://huggingface.co/cointegrated/rubert-tiny)

**For train.py:**
```
input: python train.py train.txt
output: Epoch Epoch 1/1
Train loss 0.3909 accuracy 0.8618
Val loss 0.3778 accuracy 0.9021
```
**For generate.py:**
```
input: python test.py model.pth test.txt
output: Accuracy: 0.9071
```

