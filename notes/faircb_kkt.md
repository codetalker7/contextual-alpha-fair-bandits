# KKT Conditions for the FairCB projection problem.

The problem can be framed as follows.

$$
    \begin{aligned}
        &\text{Minimize:} \sum_{s = 1}^{t - 1}\sum_{i\in[N]} p^{j_s}_i \hat{l}_{s, i} + \dfrac{1}{\eta}\sum_{j\in[M]}\sum_{i\in[N]} p^j_i \ln p^j_i\\
        &\text{Subject to:} -p^j_i\le 0\quad \forall i\in[N], j\in[M]\\
        &\quad\quad\quad\quad\quad -\sum_{j\in[M]}q_jp^{j}_i + v\le 0\quad \forall i\in[N]\\
        &\quad\quad\quad\quad\quad \sum_{i\in[N]}p^{j}_i - 1= 0\quad \forall j\in[M]\\
    \end{aligned}
$$

Let $a_{ij}$ be the Lagrange multipliers for the inequality constraint of the first type; $c_{i}$ be the ones for the inequality constraint of the second type, and let $d_j$ be the ones for the equality constraint. Then, the Lagrangian is given by the following.

$$
    \begin{aligned}
        L(\bm{p}, \bm{a}, \bm{c}, \bm{d}) = \sum_{s = 1}^{t - 1}\sum_{i\in[N]} p^{j_s}_i \hat{l}_{s, i} + \dfrac{1}{\eta}\sum_{j\in[M]}\sum_{i\in[N]} p^j_i \ln p^j_i + \sum_{ij}a_{ij}(-p^j_i) + \sum_{i}c_i\left(-\sum_{j\in[M]}q_jp^{j}_i + v\right) + \sum_{j}d_j\left(\sum_{i\in[N]}p^{j}_i - 1\right)
    \end{aligned}
$$

Next, the KKT conditions for this problem are given by the following.

1. $p^j_i\ge 0$ for all $j, i$.
2. $v\le \sum_{j}q_jp^j_i$ for all $i$.
3. $\sum_i p^j_i = 1$.
4. $a_{ij}\ge 0$ and $c_i\ge 0$.
5. $a_{ij}(-p^j_i) = 0$.
6. $c_i\left(-\sum_{j\in[M]}q_jp^j_i + v\right) = 0$.
7. $\frac{\partial L}{\partial p^j_i} = 0$.

Now, let $i'\in[N], j'\in[M]$. Condition 7. above implies the following equation.

$$
    \begin{aligned}
        \sum_{s:j_s = j'} \hat{l}_{s, i'} + \dfrac{1}{\eta} (\ln p^{j'}_{i'} + 1) - a_{j'i'} + c_{i'}(-q_{j'}) + d_{j'} = 0
    \end{aligned}
$$
