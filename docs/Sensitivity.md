# Sensitivity Analysis

## Final hidden state wrt earlier hidden states

How much change is there in the final hidden layer when you perturb an earlier hidden layer?

### Perturbed hidden state

Perturbation hidden states by multiplying by  `1 + (rand(0,1)-0.5)*0.01`

#### Table 1

|perturb hidden states before layer|MSE loss at the end|MSE loss at the end|MSE loss at the end|
|-|-|-|-|
|states perturbed|img|txt|x|
|  0 |  33.0890 +/-   3.7874|  33.7907 +/-   3.8527||
|  1 |  29.1840 +/-   3.1784|  34.3451 +/-   3.9801||
|  2 |  26.1842 +/-   3.0063|  35.6959 +/-   4.1411||
|  3 |  24.9846 +/-   2.9218|  35.9065 +/-   4.1854||
|  4 |  24.6515 +/-   2.7708|  36.6737 +/-   4.3747||
|  5 |  22.8256 +/-   2.6573|  37.1647 +/-   4.3950||
|  6 |  22.3125 +/-   2.5129|  34.4390 +/-   4.1094||
|  7 |  22.7619 +/-   2.6340|  35.4598 +/-   4.1795||
|  8 |  20.7475 +/-   2.4565|  35.4144 +/-   4.3268||
|  9 |  18.6719 +/-   2.1442|  35.5865 +/-   4.3530||
| 10 |  17.0324 +/-   1.9751|  34.2712 +/-   4.1779||
| 11 |  14.6859 +/-   1.7392|  36.1818 +/-   4.5871||
| 12 |  13.2970 +/-   1.5462|  35.3586 +/-   4.3910||
| 13 |  12.7398 +/-   1.5240|  38.8503 +/-   4.8714||
| 14 |  10.5542 +/-   1.1927|  38.4359 +/-   4.7852||
| 15 |   9.4681 +/-   1.1040|  30.6114 +/-   3.7972||
| 16 |   7.9387 +/-   0.9049|  29.5301 +/-   3.7334||
| 17 |   7.1108 +/-   0.8546|  28.0189 +/-   3.5853||
| 18 |   4.8908 +/-   0.5265|  26.4334 +/-   3.4645||
| 19 |||  13.6130 +/-   1.7506|
| 20 |||   9.8836 +/-   1.2649|
| 21 |||   7.7612 +/-   0.9777|
| 22 |||   5.9002 +/-   0.6620|
| 23 |||   4.1810 +/-   0.4079|
| 24 |||   3.0503 +/-   0.2319|
| 25 |||   2.3322 +/-   0.1323|
| 26 |||   2.0017 +/-   0.0830|
| 27 |||   1.9152 +/-   0.0743|
| 28 |||   1.7724 +/-   0.0608|
| 29 |||   1.7068 +/-   0.0549|
| 30 |||   1.6883 +/-   0.0548|
| 31 |||   1.6582 +/-   0.0519|
| 32 |||   1.5941 +/-   0.0522|
| 33 |||   1.5903 +/-   0.0499|
| 34 |||   1.5733 +/-   0.0495|
| 35 |||   1.6259 +/-   0.0533|
| 36 |||   1.5572 +/-   0.0495|
| 37 |||   1.6211 +/-   0.0537|
| 38 |||   1.5785 +/-   0.0477|
| 39 |||   1.5796 +/-   0.0508|
| 40 |||   1.5575 +/-   0.0469|
| 41 |||   1.5647 +/-   0.0507|
| 42 |||   1.5727 +/-   0.0525|
| 43 |||   1.5733 +/-   0.0502|
| 44 |||   1.5449 +/-   0.0500|
| 45 |||   1.5351 +/-   0.0500|
| 46 |||   1.6021 +/-   0.0542|
| 47 |||   1.5817 +/-   0.0549|
| 48 |||   1.5283 +/-   0.0546|
| 49 |||   1.5052 +/-   0.0488|
| 50 |||   1.5470 +/-   0.0502|
| 51 |||   1.5155 +/-   0.0541|
| 52 |||   1.4063 +/-   0.0503|
| 53 |||   1.3161 +/-   0.0530|
| 54 |||   1.2130 +/-   0.0566|
| 55 |||   1.0453 +/-   0.0505|
| 56 |||   0.8588 +/-   0.0386|

## Magnitude of components

Perturbation more generally is `dX = X * ((rand(0,1)-0.5)*delta)`.

If a tensor `X` has perturbation `dX`,
the loss in X (`MSELoss(X,(X+dX))`) is proportional to the MS value of X, `MSELoss(X,0)`

`Loss ~= delta^2 . MS(X) / 12` 

#### Table 2

|layer|MS(img)|MS(txt)|MS(x)|
|-|-|-|-|
|0|0.2379185267857143|0.86640625||
|1|3.9419642857142856|16.07857142857143||
|2|159.14285714285714|59.425||
|3|432.34285714285716|77.43571428571428||
|4|468.6857142857143|89.67857142857143||
|5|471.2857142857143|98.46428571428571||
|6|464.51428571428573|107.60714285714286||
|7|457.8857142857143|118.36428571428571||
|8|447.8857142857143|129.54285714285714||
|9|444.51428571428573|143.3||
|10|444.45714285714286|152.67142857142858||
|11|459.54285714285714|317.6||
|12|472.62857142857143|341.54285714285714||
|13|485.37142857142857|669.3142857142857||
|14|494.85714285714283|786.8571428571429||
|15|537.7714285714286|1230.9714285714285||
|16|517.4285714285714|1324.3428571428572||
|17|551.4285714285714|1544.8||
|18|578.9714285714285|1907.4285714285713||
|19|||21665.82857142857|
|20|||21818.057142857142|
|21|||22009.6|
|22|||22292.571428571428|
|23|||22591.542857142857|
|24|||22879.542857142857|
|25|||22991.085714285713|
|26|||23223.314285714285|
|27|||23763.2|
|28|||24159.085714285713|
|29|||24648.22857142857|
|30|||25391.542857142857|
|31|||26333.257142857143|
|32|||27038.17142857143|
|33|||27914.97142857143|
|34|||28138.057142857142|
|35|||28611.657142857144|
|36|||28104.22857142857|
|37|||28943.542857142857|
|38|||30182.4|
|39|||30598.4|
|40|||31172.571428571428|
|41|||31445.942857142858|
|42|||32617.14285714286|
|43|||33869.71428571428|
|44|||34943.08571428571|
|45|||37085.25714285715|
|46|||39703.77142857143|
|47|||42219.885714285716|
|48|||43724.8|
|49|||45107.2|
|50|||46101.94285714286|
|51|||46447.54285714286|
|52|||46581.02857142857|
|53|||45438.171428571426|
|54|||44768.91428571429|
|55|||44756.114285714284|
|56|||42934.857142857145|

So, if a mean square loss `L` is introduced by an approximation, it is equivalent to a delta
of `delta = sqrt(12*L/R)`, where `R` is the RMS value of layer into which the 
loss was introduced (Table 2).

The impact of delta on the final hidden state is `IMPACT = (100*delta)S`, 
where `S` is the sensitivity of the final hidden state to the one perturbed (Table 1).

So `IMPACT ~ S.sqrt(L/R)`

We can thus create a table of relative importance, `S=1`, `I = 100*sqrt(12)*sqrt(L/R)`

For the impact of an error on the final layer

## Relative Importance

Impact on the final result of a mean square loss of L is `sqrt(L).Table3`

```python
from modules.sensitivity import Sensitivity
impact = Sensitivity.impact_of_mseloss(mse_loss=loss(x,y), before_layer=n, element=Sensitivity.[IMG|TXT|X])
```

#### Table 3

|before layer|img|txt|x|
|-|-|-|-|
|0|67.83744951|36.30248882||
|1|14.69902315|8.565269959||
|2|2.075609903|4.630562401||
|3|1.201595399|4.080397002||
|4|1.138682097|3.872669074||
|5|1.051429274|3.745340131||
|6|1.03525824|3.319938068||
|7|1.063726464|3.25931204||
|8|0.980352291|3.111523981||
|9|0.885616402|2.972775966||
|10|0.807906129|2.773642139||
|11|0.685074112|2.030251882||
|12|0.611636906|1.91325324||
|13|0.578263135|1.501686683||
|14|0.47444448|1.370216235||
|15|0.408285516|0.872487485||
|16|0.348999004|0.811455486||
|17|0.302812523|0.712877571||
|18|0.203259727|0.605241761||
|19|||0.092483912|
|20|||0.066912479|
|21|||0.052314587|
|22|||0.039517244|
|23|||0.027816803|
|24|||0.020165959|
|25|||0.015381052|
|26|||0.013135206|
|27|||0.012424007|
|28|||0.011403062|
|29|||0.010871508|
|30|||0.0105951|
|31|||0.010218441|
|32|||0.009694533|
|33|||0.009518324|
|34|||0.009379172|
|35|||0.00961219|
|36|||0.009288778|
|37|||0.009528707|
|38|||0.009085895|
|39|||0.009030208|
|40|||0.008821486|
|41|||0.00882366|
|42|||0.00870809|
|43|||0.008548812|
|44|||0.00826456|
|45|||0.007971426|
|46|||0.008040328|
|47|||0.007697782|
|48|||0.007308777|
|49|||0.007087144|
|50|||0.007204945|
|51|||0.00703193|
|52|||0.006515885|
|53|||0.006174168|
|54|||0.005732876|
|55|||0.004940999|
|56|||0.004144642|
|after 56|||0.004232011|