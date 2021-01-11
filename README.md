### multeacher knowledge distillation

env: python3.6

```python

pip install -r requirements.txt

```

run demo:

```
--teacher_type gpt2 --student_type cnn --batch_size 16
```

teacher type: bert roberta gpt2 rAg(roberta+gpt2)
student_type: cnn transform