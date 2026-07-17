# GDIS mathematics

pyGDIS implements the reference formulation described in the companion manuscript.

## Descriptor transformation

The raw principal descriptors `J`, `S`, and `A` are logarithmically compressed, robustly scaled by their 5th and 95th percentiles, clipped to `[0, 1]`, and exponentially saturated.

The robust normalization operator is

$$
\widehat D=\operatorname{clip}\left(\frac{D-P_5(D)}{P_{95}(D)-P_5(D)},0,1\right).
$$

It is invariant to positive affine transformations of its input descriptor.

## Core sustained-instability measure

$$
C=\left(\widetilde J^{\alpha_J}\widetilde S^{\alpha_S}\widetilde A^{\alpha_A}\right)^{1/(\alpha_J+\alpha_S+\alpha_A)},
$$

with the fixed reference exponents

$$
\alpha_J=0.42,\qquad \alpha_S=0.33,\qquad \alpha_A=0.25.
$$

Complexity and persistence modulate the core:

$$
X=C(1+\eta_H\widehat H)(1+\eta_T\widehat T).
$$

The sustained-instability functional is obtained using Hill saturation:

$$
I_{\mathrm{sustained}}=\frac{X^\gamma}{X^\gamma+c}.
$$

## Transition localization

The normalized transition energy is based on descriptor variation over the ordered control parameter:

$$
E_{\mathrm{transition}}=\sqrt{\left|\frac{d\widehat J}{dp}\right|^2+\left|\frac{d\widehat S}{dp}\right|^2+\left|\frac{d\widehat A}{dp}\right|^2}.
$$

A Gaussian window localizes transition activity around a supplied or estimated critical parameter:

$$
W(p)=\exp\left[-\frac{1}{2}\left(\frac{p-p_c}{\sigma_p}\right)^2\right].
$$

pyGDIS stores the unweighted term

$$
I_{\mathrm{transition,base}}=E_{\mathrm{transition}}W(p)
$$

and applies the configurable transition weight `lambda_t` only at final aggregation.

## Instability potential and score

$$
\Phi=-\ln(1-I_{\mathrm{sustained}})+\lambda_t I_{\mathrm{transition,base}},
$$

$$
\mathrm{GDIS}=1-e^{-\Phi},\qquad 0\leq\mathrm{GDIS}<1.
$$

When `lambda_t = 0`, the mapping is baseline preserving:

$$
\mathrm{GDIS}=I_{\mathrm{sustained}}.
$$

The implementation defaults to `lambda_t = 0.18`, matching the manuscript reference configuration.
