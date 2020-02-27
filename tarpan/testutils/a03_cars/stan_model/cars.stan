data{
    int n;
    vector[n] dist;
    int speed[n];
}
parameters{
    real a;
    real b;
    real<lower=0> sigma;
}
model{
    vector[n] mu;
    sigma ~ exponential( 1 );
    b ~ normal( 0 , 10 );
    a ~ normal( 0 , 100 );
    for ( i in 1:n ) {
        mu[i] = a + b * speed[i];
    }
    dist ~ normal( mu , sigma );
}
generated quantities{
    vector[n] lpd_pointwise;
    {
      real mu;

      for (i in 1:n) {
          mu = a + b * speed[i];
          lpd_pointwise[i] = normal_lpdf(dist[i] | mu, sigma);
      }
    }
}
