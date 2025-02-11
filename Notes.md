# Week 1
## Discount Factor
### Types of DF curve

* Bootstrap: You have to get an squared matrix (each bond has a different maturity), therefore, the applied filters are:

    - Positive YTM `RESTRICT_YLD` 
    - Filter out TIPs (inflation linked bonds) `RESTRICT_TIPS`
    - Filter maturity dates, this means only columns with CF >= 100 and check that each bond (row) has all the CF (ie don't leave a bond CF without a coupon or nominal flow) `RESTRICT_DTS_MATURING`
    - Drop duplicated time to maturity `RESTRICT_REDUNDANT`
    
* OLS: You have redundant bonds, ie in a specific Maturity you have more than one bond that pay nominal:

    - Positive YTM `RESTRICT_YLD` 
    - Filter out TIPs (inflation linked bonds) `RESTRICT_TIPS`
    - Filter maturity dates, this means only columns with CF >= 100 and check that each bond (row) has all the CF (ie don't leave a bond CF without a coupon or nominal flow) `RESTRICT_DTS_MATURING`


### Curves

- Yield curve: Pricing a particular bond (Treasury) using the Yield to Maturity( YTM), for a particular security (Plot: YTM/ Time to Maturity (TTM)) -> It can be used as a discount rate **only** that specific bond

- Zero coupon bond: Usually there is not in the market, it is useful to create the discount rate directly using its prices (STRIS)

- Spot curve: Approach using bonds in the market to get the spot discount curve
 
# Week 2

## Sensitivity

- First approach: A parallet shift -> it could be either the YTM curve or Spot curve(*), it should result with the same result
    * Why don't use a regression: 
        1) We have an economic assumption/model of how the sensitivity should look like [Mathematical form]
        2) Every day through time, the bonds change in the matutirity (1Y -> 6M) [How to treat underlying data]

**Taylor's Approximation**

From a Taylor's approximation to the second order, we have a percentage change in price, $P$ as
$$\begin{align}
\frac{dP}{P} \approx -D\times dr + \frac{1}{2}C\times (dr)^2
\end{align}$$
where 
* $dr$ is a small change in the level of the spot curve
* $D$ is the **duration**
* $C$ is the **convexity**

(*) With this approach, it's usually refer to Spot Curve (dr)

### Duration

**Duration** refers to the sensitivity of a bond (or other fixed-income product) to the **level of interest rates**.

- The alternative perspective, it's the average weight of the cashflow maturitues

$$\begin{align}
D \equiv -\frac{1}{P}\frac{dP}{dr}
\end{align}$$

For a **zero coupon bond** this derivative has a simple solution:

$\displaystyle\text{discount} \equiv \; Z(t,T) = \frac{1}{\left(1+\frac{r_n}{n}\right)^{n(T-t)}} \; = e^{-r(T-t)}$

$$\begin{align}
D_{\text{zero}} \equiv -\frac{1}{P(t,T,0)}\frac{dP(t,T,0)}{dr} = T-t
\end{align}$$


**For zero-coupon treasuries, duration is equal to the maturity**

Comments:

- In general, we talk about the general definition of duration, this is the derivative respect to dr of the price and not as the averege cashflow 
- The coupons reduce the Duration 



