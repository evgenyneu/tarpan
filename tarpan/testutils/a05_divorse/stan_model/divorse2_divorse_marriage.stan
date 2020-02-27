/*
  The model is borrowed from Statistical Rethinking textbook
  by Richard McElreath.
*/

data{
    int n;
    vector[n] D;
    vector[n] M;
}

parameters{
    real a;
    real bM;
    real<lower=0> sigma;
}

model{
    vector[n] mu;
    sigma ~ exponential(1);
    bM ~ normal(0, 0.5);
    a ~ normal(0, 0.2);

    for (i in 1:n) {
        mu[i] = a + bM * M[i];
    }

    D ~ normal(mu , sigma);
}

generated quantities{
    vector[n] lpd_pointwise;

    for (i in 1:n) {
        real mu = a + bM * M[i];
        lpd_pointwise[i] = normal_lpdf(D[i] | mu, sigma);
    }
}
