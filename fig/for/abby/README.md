# README

## Time series

- I chose regions based on the map of the influence of land-atmosphere feedbacks on ET from Zarakas, Swann, & Battisti (2024). The Amazon, Congo Basin, and Siberia have strong negative land-atmosphere feedbacks while the Northern Great Plains has positive land-atmosphere feedbacks.

## LAI-aridity bins

- I used the simple Budyko aridity, or dryness, index defined as the ratio of climatological net downward radiation at the surface to climatological precipitation (Rn/P). Higher values indicate greater aridity.
- I masked out glaciated and snow covered land grid cells, where I define "glaciated" as glaciated land fraction greater than 0.8 and "snow covered" as time-mean snow cover fraction greater than 0.80. These criteria essentially mask out the same grid cells since glaciated land tends to have the highest time-mean snow cover fraction.
- I pooled grid cells across space and time for each ensemble member, selecting the 20-year period from 1995-2014 for a total of 240 months.
- I computed the quantile bin edges from the full ensemble pool to guarantee identical bin definitions across members. However, this means that the "quantiles" do not have exactly equal numbers of grid cells for each member.
- I placed all bare soil grid cells (LAI = 0) into the lowest quantile bin so it has a higher sample count than the other bins. However, I believe this is still better than the bare soil grid cells being distributed randomly across multiple lower quantile bins that all have LAI = 0.
- For each bin mean, I performed a two-sided t-test against a null hypothesis of 0 at alpha=0.05. Statistically insignificant bins are masked out. This assumes i.i.d. samples within each bin, which is a potentially faulty assumption if there is strong spatial or temporal correlation between bin samples. More complex techniques such as block bootstrapping or applying a spatial decorrelation adjustment to the effective degrees of freedom could be implemented for a more robust test.
- The radius of the circles is proportional to the number of grid cell samples in the bin.
