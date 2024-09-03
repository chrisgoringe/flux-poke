# Casting cost

## General Observations

- bitsandbytes doesn't perform well
- mostly the on the fly ones work well, although Qx_K_S perform better or as well and are slightly smaller
- there is a very strong dependance on depth in the model, though not entirely monotonic
- some interesting variation in last 10 or so layers with smaller quants performing very badly
- the quantisation or not of bias (the difference between Q4_1 and Q4_1* and similar) seems to make no difference

Model - Flux.1.dev

Quantizations marked with (*) are patched in from GGUF models. Others are
quantised by this code.

In all models, the entry and exit layers are unquantised: 64,124,992 parameters

In all models, the normalisation scales are unquantised: 19,456 parameters

In patch models, block biases are unquantised: 3,035,136 parameters

|type|quantised|unquantised block biases|unquantised other|q%|
|:-:|-:|-:|-:|-:|
|fly|11,837,294,656|0|64,144,448|99.461%|
|patch|11,834,228,736|3,065,920|64,144,448|99.435%|

---

## Bits per parameter:

|-|Q8_0|bf8|bnb8|Q5_1|Q5_K_S*|Q4_1|Q4_1*|Q4_K_S*|Q4_0*|bnbFP4|bnbNF4|Q3_K_S*|Q2_K*|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|bits|8.5|8|8+|6|5.5|5|5|4.5|4.5|4+|4+|3.4375|2.625|

---

## Error (average MSE error in final hidden state) of quantising layers to different levels.

Note that layers 19-56 are single block (141,557,760 quantable parameters), 
layers 0-18 are double block (339,738,624 quantable parameters). Quantizing a double
block saves 2.4 times the memory of quantizing a single block. See weighted table below
to make comparisons between layers.


|-|Q8_0|bf8|bnb8|Q5_1|Q5_K_S*|Q4_0*|Q4_1|Q4_1*|Q4_K_S*|bnbFP4|bnbNF4|Q3_K_S*|Q2_K*|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|0| 61.301||| 79.268| 91.402|183.730|110.990|110.960|125.260|||331.730|652.650|
|1| 59.049||| 79.055| 75.232|120.780| 97.950| 95.388|112.920|||168.660|315.610|
|2| 52.052||| 67.571| 72.492|100.110| 84.105| 82.944| 91.237|||155.940|252.350|
|3| 46.173||| 57.827| 58.888| 76.786| 69.898| 70.468| 75.396|||113.890|173.610|
|4| 41.890||| 57.455| 56.013| 85.789| 73.359| 73.093| 72.959|||123.710|173.780|
|5| 40.371||| 57.594| 53.946| 77.101| 71.558| 70.572| 70.298|||105.310|167.140|
|6| 39.330||| 54.982| 54.866| 74.507| 69.990| 71.033| 69.467||| 99.927|171.260|
|7| 38.649||| 58.501| 55.181| 76.562| 73.138| 72.548| 74.795|||112.250|187.250|
|8| 39.691||| 56.240| 55.221| 74.278| 73.514| 72.145| 72.967|||104.670|176.220|
|9| 37.113||| 53.990| 55.356| 76.201| 71.753| 71.266| 72.076|||103.180|192.610|
|10| 42.300||111.025| 61.516| 63.087| 85.732| 72.594| 73.166| 78.085| 94.858| 86.440|110.230|165.180|
|11| 32.119||| 48.114|| 69.333| 64.662| 65.347| 65.917||| 97.697|162.110|
|12| 36.311||| 56.112|| 77.444| 71.314| 71.348| 73.896||| 99.562|139.030|
|13| 28.363||| 44.029|| 59.563| 60.592| 60.293| 59.012||| 93.597|144.570|
|14| 31.585||| 45.968|| 66.275| 59.777| 58.976| 56.921||| 88.855|131.790|
|15| 22.684||| 34.331|| 48.859| 51.835| 51.816| 42.946||| 72.874|129.310|
|16| 21.366||| 33.128|| 51.315| 45.384| 45.695| 50.804||| 69.755|130.860|
|17| 22.821||| 34.614|| 47.402| 47.378| 46.264| 41.880||| 71.642|147.760|
|18| 20.165||| 27.116|| 40.205| 36.467| 37.402| 39.689||| 64.070|204.030|
|19|  8.992|| 39.222| 13.120|| 19.431| 18.577| 18.433| 17.755| 26.529| 21.028| 32.293| 66.051|
|20|  7.014||| 10.336|| 17.250| 14.754| 14.848| 14.212||| 27.637| 61.560|
|21|  5.661|||  8.600|| 16.810| 14.101| 14.082| 13.667||| 29.931| 66.440|
|22|  4.301|||  6.731|| 12.997| 11.502| 11.452| 11.354||| 29.641| 65.479|
|23|  3.228|||  5.311|| 11.361|  9.433|  9.434|  9.333||| 22.168| 57.911|
|24|  2.611|||  4.398|| 10.041|  8.574|  8.444|  8.521||| 25.752| 55.281|
|25|  2.228|||  3.870||  8.861|  7.794|  7.672|  7.317||| 20.823| 53.874|
|26|  2.008|||  3.313||  7.534|  6.301|  6.298|  5.955||| 17.519| 50.330|
|27|  1.797|||  3.087||  7.004|  6.473|  6.466|  5.791||| 17.471| 51.741|
|28|  1.724|||  3.022||  7.332|  6.767|  6.749|  6.088||| 17.565| 52.947|
|29|  1.654|||  2.783||  6.877|  5.849|  5.839|  5.571||| 16.025| 51.009|
|30|  1.593|||  2.746||  6.393|  5.622|  5.601|  5.008||| 15.292| 49.979|
|31|  1.517|||  2.630||  6.464|  5.408|  5.358|  5.000||| 15.007| 53.374|
|32|  1.489|||  2.651||  6.630|  5.463|  5.453|  5.104||| 16.677| 51.676|
|33|  1.431|||  2.571||  6.073|  5.411|  5.441|  4.833||| 15.490| 51.304|
|34|  1.365|||  2.407||  5.692|  4.817|  4.817|  4.407||| 13.118| 47.748|
|35|  1.349|||  2.454||  6.654|  4.821|  4.798|  4.364||| 14.165| 44.308|
|36|  1.279|||  2.326||  5.497|  4.650|  4.646|  4.205||| 14.435| 43.085|
|37|  1.194|||  2.091||  4.593|  4.095|  4.085|  3.743||| 10.838| 37.062|
|38|  1.140|||  2.022||  4.464|  3.855|  3.820|  3.617||| 11.213| 37.229|
|39|  1.131|||  1.992||  4.867|  4.215|  4.201|  3.556||| 10.457| 36.870|
|40|  1.078|||  2.054||  5.156|  4.331|  4.323|  3.953||| 12.249| 42.524|
|41|  1.056|||  2.022||  5.035|  4.365|  4.342|  4.082||| 12.288| 45.209|
|42|  1.002|||  2.163||  5.814|  4.869|  4.841|  4.251||| 15.131| 53.958|
|43|  0.956|||  2.088||  5.937|  5.198|  5.184|  4.506||| 15.065| 57.444|
|44|  0.963|||  2.343||  7.321|  6.147|  6.115|  6.208||| 19.369| 68.677|
|45|  0.904|||  2.304||  7.115|  6.381|  6.354|  5.321||| 18.907| 70.569|
|46|  0.848|||  2.626||  9.028|  7.748|  7.692|  6.535||| 24.860| 93.890|
|47|  0.750|||  2.490||  8.525|  7.172|  7.112|  6.270||| 26.381| 97.773|
|48|  0.730|||  2.684||  9.582|  7.873|  7.829|  7.327||| 30.304|115.310|
|49|  0.668|||  3.441|| 13.770| 11.272| 11.243|  9.748||| 39.521|156.840|
|50|  0.634|||  3.585|| 15.393| 11.870| 11.802| 11.562||| 42.105|172.190|
|51|  0.650|||  4.752|| 20.733| 16.265| 16.093| 14.278||| 64.784|236.590|
|52|  0.530|||  4.236|| 20.070| 15.376| 15.308| 13.765||| 55.817|223.320|
|53|  0.585|||  5.479|| 44.167| 21.641| 21.406| 23.535|||119.310|355.650|
|54|  0.384|||  6.060|| 27.048| 22.456| 22.218| 18.431||| 92.537|322.560|
|55|  0.183|||  1.217||  5.793|  4.200|  4.094|  4.365||| 19.676| 62.069|
|56|  0.137|||  0.308||  3.901|  0.926|  0.881|  4.313||| 19.450| 74.913|

---

## Weighted

Same again, but the double block values divided by 2.4. So this is proportional to error per parameter quantized.


|-|Q8_0|bf8|bnb8|Q5_1|Q5_K_S*|Q4_0*|Q4_1|Q4_1*|Q4_K_S*|bnbFP4|bnbNF4|Q3_K_S*|Q2_K*|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|0| 25.542||| 33.028| 38.084| 76.554| 46.246| 46.233| 52.192|||138.221|271.938|
|1| 24.604||| 32.940| 31.347| 50.325| 40.812| 39.745| 47.050||| 70.275|131.504|
|2| 21.688||| 28.155| 30.205| 41.712| 35.044| 34.560| 38.015||| 64.975|105.146|
|3| 19.239||| 24.095| 24.537| 31.994| 29.124| 29.362| 31.415||| 47.454| 72.338|
|4| 17.454||| 23.940| 23.339| 35.745| 30.566| 30.455| 30.400||| 51.546| 72.408|
|5| 16.821||| 23.998| 22.477| 32.125| 29.816| 29.405| 29.291||| 43.879| 69.642|
|6| 16.387||| 22.909| 22.861| 31.045| 29.162| 29.597| 28.945||| 41.636| 71.358|
|7| 16.104||| 24.375| 22.992| 31.901| 30.474| 30.228| 31.165||| 46.771| 78.021|
|8| 16.538||| 23.433| 23.009| 30.949| 30.631| 30.060| 30.403||| 43.613| 73.425|
|9| 15.464||| 22.496| 23.065| 31.750| 29.897| 29.694| 30.032||| 42.992| 80.254|
|10| 17.625|| 46.260| 25.632| 26.286| 35.722| 30.247| 30.486| 32.535| 39.524| 36.017| 45.929| 68.825|
|11| 13.383||| 20.047|| 28.889| 26.943| 27.228| 27.465||| 40.707| 67.546|
|12| 15.130||| 23.380|| 32.268| 29.714| 29.728| 30.790||| 41.484| 57.929|
|13| 11.818||| 18.345|| 24.818| 25.247| 25.122| 24.588||| 38.999| 60.237|
|14| 13.160||| 19.153|| 27.615| 24.907| 24.573| 23.717||| 37.023| 54.913|
|15|  9.452||| 14.305|| 20.358| 21.598| 21.590| 17.894||| 30.364| 53.879|
|16|  8.902||| 13.803|| 21.381| 18.910| 19.040| 21.168||| 29.065| 54.525|
|17|  9.509||| 14.422|| 19.751| 19.741| 19.277| 17.450||| 29.851| 61.567|
|18|  8.402||| 11.298|| 16.752| 15.195| 15.584| 16.537||| 26.696| 85.013|
|19|  8.992|| 39.222| 13.120|| 19.431| 18.577| 18.433| 17.755| 26.529| 21.028| 32.293| 66.051|
|20|  7.014||| 10.336|| 17.250| 14.754| 14.848| 14.212||| 27.637| 61.560|
|21|  5.661|||  8.600|| 16.810| 14.101| 14.082| 13.667||| 29.931| 66.440|
|22|  4.301|||  6.731|| 12.997| 11.502| 11.452| 11.354||| 29.641| 65.479|
|23|  3.228|||  5.311|| 11.361|  9.433|  9.434|  9.333||| 22.168| 57.911|
|24|  2.611|||  4.398|| 10.041|  8.574|  8.444|  8.521||| 25.752| 55.281|
|25|  2.228|||  3.870||  8.861|  7.794|  7.672|  7.317||| 20.823| 53.874|
|26|  2.008|||  3.313||  7.534|  6.301|  6.298|  5.955||| 17.519| 50.330|
|27|  1.797|||  3.087||  7.004|  6.473|  6.466|  5.791||| 17.471| 51.741|
|28|  1.724|||  3.022||  7.332|  6.767|  6.749|  6.088||| 17.565| 52.947|
|29|  1.654|||  2.783||  6.877|  5.849|  5.839|  5.571||| 16.025| 51.009|
|30|  1.593|||  2.746||  6.393|  5.622|  5.601|  5.008||| 15.292| 49.979|
|31|  1.517|||  2.630||  6.464|  5.408|  5.358|  5.000||| 15.007| 53.374|
|32|  1.489|||  2.651||  6.630|  5.463|  5.453|  5.104||| 16.677| 51.676|
|33|  1.431|||  2.571||  6.073|  5.411|  5.441|  4.833||| 15.490| 51.304|
|34|  1.365|||  2.407||  5.692|  4.817|  4.817|  4.407||| 13.118| 47.748|
|35|  1.349|||  2.454||  6.654|  4.821|  4.798|  4.364||| 14.165| 44.308|
|36|  1.279|||  2.326||  5.497|  4.650|  4.646|  4.205||| 14.435| 43.085|
|37|  1.194|||  2.091||  4.593|  4.095|  4.085|  3.743||| 10.838| 37.062|
|38|  1.140|||  2.022||  4.464|  3.855|  3.820|  3.617||| 11.213| 37.229|
|39|  1.131|||  1.992||  4.867|  4.215|  4.201|  3.556||| 10.457| 36.870|
|40|  1.078|||  2.054||  5.156|  4.331|  4.323|  3.953||| 12.249| 42.524|
|41|  1.056|||  2.022||  5.035|  4.365|  4.342|  4.082||| 12.288| 45.209|
|42|  1.002|||  2.163||  5.814|  4.869|  4.841|  4.251||| 15.131| 53.958|
|43|  0.956|||  2.088||  5.937|  5.198|  5.184|  4.506||| 15.065| 57.444|
|44|  0.963|||  2.343||  7.321|  6.147|  6.115|  6.208||| 19.369| 68.677|
|45|  0.904|||  2.304||  7.115|  6.381|  6.354|  5.321||| 18.907| 70.569|
|46|  0.848|||  2.626||  9.028|  7.748|  7.692|  6.535||| 24.860| 93.890|
|47|  0.750|||  2.490||  8.525|  7.172|  7.112|  6.270||| 26.381| 97.773|
|48|  0.730|||  2.684||  9.582|  7.873|  7.829|  7.327||| 30.304|115.310|
|49|  0.668|||  3.441|| 13.770| 11.272| 11.243|  9.748||| 39.521|156.840|
|50|  0.634|||  3.585|| 15.393| 11.870| 11.802| 11.562||| 42.105|172.190|
|51|  0.650|||  4.752|| 20.733| 16.265| 16.093| 14.278||| 64.784|236.590|
|52|  0.530|||  4.236|| 20.070| 15.376| 15.308| 13.765||| 55.817|223.320|
|53|  0.585|||  5.479|| 44.167| 21.641| 21.406| 23.535|||119.310|355.650|
|54|  0.384|||  6.060|| 27.048| 22.456| 22.218| 18.431||| 92.537|322.560|
|55|  0.183|||  1.217||  5.793|  4.200|  4.094|  4.365||| 19.676| 62.069|
|56|  0.137|||  0.308||  3.901|  0.926|  0.881|  4.313||| 19.450| 74.913|

Patches from (https://huggingface.co/city96/FLUX.1-dev-gguf)
