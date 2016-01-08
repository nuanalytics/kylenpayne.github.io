---
layout: post
title: What happens to the fixed effects?
---

<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


## Introduction

I am currently working on a problem which seems rather applied in nature: What happens to the fixed effects in a mixed effects model when certain variables are treated as fixed instead of random? Let me clear up what I mean...

Let's say of the predictors that you choose for your mixed model, there is some subset of these that are fixed effects only, with design $X_{p}$.
$X$ has the form 

$$
X = 
\begin{pmatrix}
X_1 \\
\vdots \\
X_N
\end{pmatrix}
$$
Where canonically, each $X_{i}$ corresponds to the design matrix of fixed effects for the $i^{th}$ cluster. 
 
 Likewise there is a set of predictors that are treated only as random effects with design matrix $Z$, where $Z_{l}$ is block diagonal $Z_{l} = blockdiag(Z_1, ..., Z_N)$. Lastly, there is some subset of the predictors that are up for discussion. Let's assume that the rank of the design matrix $W$ for these predictors is relatively small. Then we are attempting to compare the fixed effects for the two following models.

$$
M_{0}: Y = X \beta + W \alpha + Z \gamma + \epsilon
$$

with $\beta$ fixed, $\alpha \sim N(0_{Nq}, A), \gamma \sim N(0_{Nl}, \Gamma), \epsilon \sim N(0_{N_{T}}, \sigma^2 I_{N_{T}})$ where $N_{T} = \sum_{i=1}^{N}n_{i}$


$$
M_{1}: Y = X^{*}\beta^{*} + Z\gamma + \epsilon
$$

with 
$$ 
X^{*} = (X, WL)
$$,
 
$$
\beta^{*} = (\beta_{1}, \omega)^{T}
$$,

$$
L = 
\begin{pmatrix}
I_q \\
\vdots \\
I_q
\end{pmatrix}
$$

with dimension $Nq \times q$. $L$ flattens the block diagonal version of $W$ into a population averaged version. I.e. if 

$$
W = 

\begin{pmatrix}
W_1 & 0 & \dots & 0 \\
0 & W_2 & \dots & 0 \\
0 & 0 & \ddots & 0 \\
0 & 0 & 0 & W_N
\end{pmatrix}
$$

Then 
$$
WL = 
\begin{pmatrix}
W_1 \\
\vdots \\
W_N
\end{pmatrix}
$$

So as you can probably tell by now, I want to know what happens to the hypothesis tests for the $\beta$ and $\beta_1$ when the $W$ variables to 
be either fixed or random effects. This question has come as a consequence of fitting models to electrophysiological data, namely, averaged EEG data. 
Due to the central limit theorem, the response $Y$, average amplitude of the electrical signal is guaranteed to be normal. Electrode label is a factor determining the location of the electrical response, and is the $W$ that inspired this line of work. This would result in a mostly sparse $W$ or $WL$ in either
instance, where the non-zero entries are just identity for all entries which coincide with some electrode position. While there are instances of treating electrode position as a fixed effect, it seemingly makes more sense to me to treat it as random. My reasoning is that while typically the positions of the electrodes are fixed on the EEG caps, the sample space of the electrode postions (i.e. the head) is not discrete. We do not exhaust all possible positions of 
electrode site. We are probably closer to taking a random sample of electrode positions on the skull. Moreover the interpretation for the parameter estimates
of $\omega$ is messy at best, as we would have to pick some electrode position to be the reference level, and thus the inferences of $\hat{\omega}$ would have to be made relative to that level. 

Moreover if we treat electrode position as random, we more or less avoid these issues. The random effects are treated as nuissance parameters and are integrated out of the likelihood. We also get the interpretation of electrode site contributing to the variance structure of the model, which would make sense in the case that observations are spatially correlated depending on the electrode site. 

## Relative Efficiency

If we use maximum likelihood to estimate either model, we end up with 
$$
\hat{\beta}_{GLS} = (X^{T}V^{-1}X)^{-1}X^{T}V^{-1}Y
$$
when $V$ is held constant. 
One result I am trying to establish is under what conditions is the Wald test of the fixed effects in model $M_{0}$ is asymptotically relatively efficient to $M_{1}$. While there is a large issue of determining the distribution of $t$ test for fixed effects in mixed models, (see note 1), there is no controversy in terms of determining the asymptotic relative efficiency. We proceed in the fashion of Lehman and Romano. 

### Theorem 1.
Consider two competing sequences of testing sequences $\{\phi\}, \{ \hat{\phi }\}$, based on test statistics $T_{n}$, and $\hat{T_{n}}$. Fix $0 < \alpha < \beta < 1$, and let $\beta{j} > \beta_{j0}$, then let $N(\beta_j)$ and $\hat{N(\beta_{j0})}$ be the smallest sample sizes necessary for $\phi$ and $\hat{\phi}$ to have
power of $\beta$ against $\beta_{j0}$. Then 

$$
lim_{\beta_j \rightarrow \beta_{j0}} \frac{N(\beta_j)}{\hat{N(\beta_{j0})}} = (\frac{\hat{(d\mu/d\beta)/\sigma}}{(d\mu/d\beta)/\sigma})^2
$$
the right hand side is called the (Pitman) Asymptotic Relative Efficiency. 

### Covariance Matrices.  

One result to compare the relative efficiency of the two models' t-test for corresponding effects 
$$
\beta_{j}, \beta^{*}_{j}
$$

This is the bulk of what I'm trying to establish in this work. Moreover, I am attempting to show this for the case that 
both models are misspecified, and the true error distribution of $\epsilon$ is $\Sigma$. 


Therefore, let 

$$
\hat{\beta}_{GLS} = (X^{T}V_1^{-1}X)^{-1}X^{T}V_1^{-1}Y
$$

be the fixed effects estimator for model $M_0$. And likewise for $M_1$
$$
\hat{\beta^{*}}_{GLS} = (X^{*T}V_2^{-1}X^{*})^{-1}X^{*T}V_2^{-1}Y
$$

Then under the assumption $Y_{N_{T}\times 1} \sim N(\mu, \Sigma)$, with $\mu \approx X\beta $, then the results of
theorem 1 hold. There are alot of steps being skipped here, but essentially, the relative efficiency of the t-test is
equivalent to the asymptotic efficiency of the effect estimates. 

We can find the variance matrices by applying the variance function, which results in

$$
V_1 = Z\Gamma Z^T + W E(\alpha \gamma^T) Z^{T} + Z E(\gamma \alpha^T) W^T + W A W^{T} + I_{N_{T}}\sigma^2
$$
which under the simplifying assumption that $\alpha \perp \gamma$ we get

$$
V_1 =  Z\Gamma Z^T  + W A W^{T} + I_{N_{T}}\sigma^2
$$

$$
V_2  = Z\Gamma Z^T + I_{N_{T}}\sigma^2
$$

We can find the variance of the parameter estimates 
$\hat{\beta}_{GLS}$ 
and 
$$
\hat{\beta^{*}}_{GLS}
$$
which results in 

$$
 Cov(\hat{\beta}_{GLS}) = (X^{T}V_1^{-1}X)^{-1}X^{T}V_1^{-1} Cov(Y) V_1^{-1} X^{T} (X^{T}V_1^{-1}X)^{-1} =  (X^{T}V_1^{-1}X)^{-1}X^{T}V_1^{-1}  \Sigma  V_1^{-1}  X^{T} (X^{T}V_1^{-1}X)^{-1}
$$ 

and 

$$
Cov(\hat{\beta^{*}}_{GLS} ) = (X^{*T}V_2^{-1}X^{*})^{-1}X^{*T}V_2^{-1} \Sigma V_2^{-1} X^{*T} (X^{*T}V_2^{-1}X^{*})^{-1}
$$

We establish asymptotic relative efficiency of the wald $t$ test if the difference in the covariance matricies

$$
Cov(\hat{\beta^{*}}_{1} ) - Var(\hat{\beta}_{GLS}) 
$$

is postive definite. This is sufficient, but not nessecary condition for the test of the $j^{th}$ fixed effect estimate for all point one sided 
null hypotheses. 

### Some Technical Results

#### The following is a set of some boring technical results.
The term 
$$
(X^{*T}V_2^{-1}X^{*})^{-1}
$$
bookends the equation for the variance of the variance of $M_1$. 
Using block inversion, we find that

$$
(X^{*T}V_2^{-1}X^{*})^{-1} = 

\begin{pmatrix}
X^T V_2^{-1}X & X^{T}V_{2}^{-1} W L \\
L^T W^T V_2^{-1} X & L^T W^T V_{2}^{-1} W L 
\end{pmatrix}^{-1}

=
\begin{pmatrix}
U_1 & U_2 \\
U_3 & U_4
\end{pmatrix}
$$

The $U_{i}'s$ are found by block inversion. But as we are only taking the first $p$ entries of 
$$
\hat{\beta^{*}}
$$

, then we only need
$U_1, U_2$ as the others get multiplied by 0 by applying $(I_p, 0)\hat{\beta^{*}}$ to take the first $p$ entries that correspond to the
same fixed effects. 

$$
U_1 = (X^T V_2^{-1}X - X^{T}V_{2}^{-1} W L (L^T W^T V_{2}^{-1} W L )^{-1}L^T W^T V_2^{-1} X)^{-1}
$$

and 

$$
U_2 = -(X^T V_2^{-1}X)^{-1} X^{T}V_{2}^{-1} W L (L^T W^T V_{2}^{-1} W L - L^T W^T V_2^{-1} X (X^T V_2^{-1}X)^{-1}  X^{T}V_{2}^{-1} W L  )^{-1}
$$

## If only everything is instantly positive definite 

$$
Cov(\hat{\beta^{*}}_1) = U_1X^{T}\Sigma X U_1 + U_2 L^{T}W^{T} \Sigma X U_{1} + U_{1}X^{T}\Sigma W L U_{2}^{T} + U_{2}L^{T}W^{T}\Sigma W L U_2^{T}
$$

and 

$$
Cov(\hat{\beta}) = (X^{T}V_1^{-1}X)^{-1}X^{T}V_1^{-1}  \Sigma  V_1^{-1}  X^{T} (X^{T}V_1^{-1}X)^{-1}
$$

We can see an obvious difference between the two matricies. In $M_{2}$ we have a fixed effects parameter that has a much more complicated mean structure,
whereas for $M_1$ we end up with a more complicated covariance structure. There may be a large set of cases in which the
