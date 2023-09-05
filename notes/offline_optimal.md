# Finding the offline optimal collection of distributions.

As usual, let $M$ be the number of contexts, and let $N$ be the number of arms. To find the best offline distribution, we need to solve the following convex optimization problem: where $\bm{r}_1, ..., \bm{r}_T$ is the reward sequence, and $(\bm{x}^1, ..., \bm{x}^M)$ is the variable collection of distributions:

$$
    \begin{aligned}
        & \text{Minimize}: -\sum_{i = 1}^N \dfrac{\left(1 + \sum_{t = 1}^T x^{j_t}_i r_i(t)\right)^{1 - \alpha}}{1 - \alpha}\\
        & \text{Subject to:} -x^j_i\le 0 \quad \forall i\in[N], j\in[M]\\
        & \quad\quad\quad\quad\quad x^j_i - 1\le 0 \quad \forall i\in[N], j\in[M]\\
        & \quad\quad\quad\quad\quad \sum_{i\in[N]}x^{j}_i - 1 = 0\quad \forall j\in[M] 
    \end{aligned}
$$

For this convex optimization problem, let $a_{ij}$ for $i\in[N], j\in[M]$ be the Lagrange multipliers for the inequality constraints of the first type; let $b_{ij}$ for $i\in[N], j\in[M]$ be the Lagrange multipliers for the inequality constraints of the second type, and let $c_j$ for $j\in[M]$ be the Lagrange multipliers for the equality constraints. Then, the Lagrangian is given by the following:

$$
    \begin{aligned}
        L(\bm{x}, \bm{a}, \bm{b}, \bm{c}) = -\sum_{i = 1}^N \dfrac{\left(1 + \sum_{t = 1}^T x^{j_t}_i r_i(t)\right)^{1 - \alpha}}{1 - \alpha} + \sum_{i, j}a_{ij}(-x^{j}_i) + \sum_{i, j}b_{ij}(x^{j}_i - 1) + \sum_{j}c_j\left(\sum_{i\in[N]}x^{j}_i - 1\right)
    \end{aligned}
$$

The KKT conditions for this problem are the following.

1. $0\le x^j_i\le 1$ for all $i\in[N], j\in[M]$.
2. $\sum_{i\in[N]}x^j_i - 1 = 0$ for all $j\in[M]$.
3. $a_{ij} \ge 0$ and $b_{ij}\ge 0$ for all $i\in[N], j\in[M]$.
4. $a_{ij}(-x^{j}_i) = 0$ for all $i\in[N], j\in[M]$.
5. $b_{ij}(x^{j}_i - 1) = 0$ for all $i\in[N], j\in[M]$.
6. $\frac{\partial L}{\partial x^{j'}_{i'}} = 0$ for all $i'\in[N], j'\in[M]$.

Now, let $i'\in[N], j'\in[M]$. Condition 6. above gives us the following equation.

$$
    \begin{aligned}
        -\left(\sum_{t:j_t = j'}r_{i'}(t)\right)\left(1 + \sum_{t = 1}^T x^{j_t}_{i'}r_{i'}(t)\right)^{-\alpha} - a_{i'j'} + b_{i'j'} + c_j = 0
    \end{aligned}
$$

Now, from conditions 4. and 5., we see that if $x^{j'}_{i'}\notin\{0, 1\}$ then $a_{i'j'} = b_{i'j'} = 0$. In that case, the last equation becomes the following.

$$
    \begin{aligned}
        c_j = \left(\sum_{t:j_t = j'}r_{i'}(t)\right)\left(1 + \sum_{t = 1}^T x^{j_t}_{i'}r_{i'}(t)\right)^{-\alpha}
    \end{aligned}
$$

Any clean way to proceed from here?
