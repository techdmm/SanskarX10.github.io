# Basics of Distillation

* paper link : [1503.02531 (arxiv.org)](https://arxiv.org/pdf/1503.02531)
- a large trained model + a good regularizer
- distillation - transfer its knowledge to the small model
- large model assigns probabilities to the wrong class, less in value but still assigns but it tells us about how they generalize
- objective_function(training) should reflect true objective , rather trained to optimize performance 
- real objective = generalize to new data
- but this needs info about correct ways to generalize , usually unavailable
- small model + large trained model data < same generalization as large model + small model
- how to do it? during knowledge transfer use soft target to train the smaller model
- large model = small model 1 + small model 2 + small model 3 
- soft target =  mean {probability_prediction (large model)} = probability_prediction (small model 1) + probability_prediction ( small_ model n)
-  $$
 H(Soft Targets)↑⟹I(Soft Targets)↑⟹σ2(Soft Targets)↓⟹Ndata​↓
$$
* H = entropy, I = information, mu = variance = N = training data (per training case)
* Higher entropy means the predictions are less confident (more spread out among different classes) but carry more information. For example, a soft target distribution like (0.6, 0.3, 0.1) has higher entropy than a hard target like (1, 0, 0)
* $$
\ softmax(q_i) = \frac{e^{z_i / T}}{\sum_{j} e
^{z_j / T}}
​​$$
*  raising the temperature in softmax gives less confident spread out probabilities  , high temp, lower logits , smaller values , softer probability  and vice versa
* small model train dataset = distilled model dataset
* distilled model dataset = transfer set + soft target distribution for each case in set
* if correct labels are known for transfer set then for significant improvement :
	* first objective function : cross entropy over soft target with high temperature softmax in distilled model
	* second objective function: cross entropy with correct labels over same logits with normal softmax
	* take weighted average of them , with lower weight on second function
	* it's important to multiply soft target function result with T^2 ( only when using both)

* $$
 \begin{equation}
\frac{\partial C}{\partial z_i} = \frac{1}{T} (q_i - p_i) = \frac{1}{T} \left( \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} - \frac{e^{v_i/T}}{\sum_j e^{v_j/T}} \right)
\end{equation}
$$
$$\begin{equation}
\frac{\partial C}{\partial z_i} \approx \frac{1}{T} \left( \frac{1 + z_i/T}{N + \sum_j z_j/T} - \frac{1 + v_i/T}{N + \sum_j v_j/T} \right)
\end{equation}
$$
$$
\begin{equation}
\frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2} (z_i - v_i)
\end{equation}$$


* Equation 1 is gradient when , Large model logits are given by vi, distilled model logits zi, soft target probabilities pi produced by vi at Temperature T
* Equation 2 is gradient when T larger then logits in magnitude
* Equation 3 is when logits are zero-meaned ( sum of mean == 0)
 
```
e = 2.718
T = 1.5
N = 4

zi = [2.34, 1.64, 4.54, 3.44]
exp_zi = [e**(i / T) for i in zi]
qi = [exp_i / sum(exp_zi) for exp_i in exp_zi]

vi = [0.06,0.003,1,0.6]
exp_vi = [e**(j / T) for j in vi]
pi = [exp_v / sum(exp_vi) for exp_v in exp_vi]
grad_vec = [(1/T)*(i - j) for i,j in zip(qi,pi)]

# now when T == 5, high temp

Th = 5

qi_ht = [(1 + (i/ Th))/(N + (sum(zi)/Th)) for i in zi]
pi_ht = [(1 + (j/ Th))/(N + (sum(vi)/Th)) for j in vi]
grad_vec_ht = [(1/T)*(i - j) for i,j in zip(qi_ht,pi_ht)]


## now when logits are zero meaned

zi_zm = [i - (sum(zi)/N) for i in zi]
vi_zm = [j - (sum(vi)/N) for j in vi]
grad_vec_zm = [ (1/(N * (T**2))) * (i - j) for i,j in zip(zi, vi)]


```

*  when model is too small to capture all knowledge from larger model , medium temperatures work file , which implies that ignoring large negative logits can be helpful

More details to be added...
