/*
  The model is borrowed from Statistical Rethinking textbook
  by Richard McElreath.
*/
data{
    int n;
    vector[n] h1;
    vector[n] h0;
}

parameters{
    real<lower=0> p;
    real<lower=0> sigma;
}

model{
    vector[n] mu;
    sigma ~ exponential(1);
    p ~ lognormal(0, 0.25);
    for (i in 1:n){
        mu[i] = h0[i] * p;
    }
    h1 ~ normal(mu, sigma);
}

generated quantities{
    vector[n] lpd_pointwise;

    for ( i in 1:n ) {
        real mu = h0[i] * p;
        lpd_pointwise[i] = normal_lpdf(h1[i] | mu, sigma);
    }
}
