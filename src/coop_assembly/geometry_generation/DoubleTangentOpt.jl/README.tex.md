

## Given data
- $G = \langle V, E \rangle$
- $e = \langle v, v' \rangle$
- $\vec{p}_v \in \mathbf{R}^3$ (given graph vertex embedding)

## Decision variables
- $\vec{x}_{e, v} \in \mathbf{R}^3$ for $v \in e$ ($e$'s end point $v$)
- $z_{e, v, e'} \in \{0, 1\}$ for $e \neq e', v \in (e \cap e')$ (tangent assignment)

## Deduced variables:
- $\vec{d}_{e, v, e'} \in \R^3$ for $e \neq e', v \in (e \cap e')$ 

    ($\vec{d}$ is the distance vector between bars $e, e'$ at the vertex $v$)

> ? Do for $d_{e, v, e', v'}$ instead? 

## Optimization formulation

$$
\begin{align*}
    \min\limits_{\mathbf{\vec{x}}, \mathbf{\vec{d}}, \mathbf{z}} &\sum\limits_{e \in E} \sum\limits_{v \in e} \| \vec{x}_{e, v} - \vec{p}_v \| \\
    \textrm{s.t. } & \vec{d}_{e, v, e'} = \vec{x}_{e, v} - \vec{x}_{e', v} \\ % & \forall e, e' \in E, v \in (e \cap e') \\
    & \norm{\vec{d}_{e, v, e'}} \geq (2r)^2 \\
    & \norm{\vec{d}_{e, v, e'}} \leq (2r + \epsilon)^2 + Mz_{e, v, e'} \\
    & \sum_{e' \in E} z_{e, v, e'} = \min{(\deg{(v)}{-}1, 2)} \\ % & \forall e \in E, \forall v \in e \\
    & \vec{x}_{e, v}, \vec{d}_{e, v} \in \R^3 \\
    & z_{e, v, e'} \in \{0, 1\}
    % & z_{e, v, e'} = 0 & v \notin (e \cap e')
\end{align*}
$$
