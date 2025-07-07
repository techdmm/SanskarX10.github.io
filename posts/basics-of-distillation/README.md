# Basics of Distillation

Distillation transfers the knowledge from a larger model to a smaller one while keeping the smaller model well regularized.

The large model assigns probabilities even to wrong classes. This behaviour gives clues about its generalization.

Training objectives should reflect the real goal of good generalization rather than just immediate performance.

$$
L = -\sum_i p_i \log q_i
$$

```python
def distill(student, teacher):
    return teacher.predict(student.inputs)
```

More details to be added...
