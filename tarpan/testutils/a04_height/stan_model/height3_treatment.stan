data{
    int n;
    vector[n] h1;
    vector[n] h0;
    int treatment[n];
}

parameters{
    real<lower=0> a;
    real bt;
    real<lower=0> sigma;
}

model{
    vector[n] mu;
    sigma ~ exponential(1);
    bt ~ normal(0, 0.5);
    a ~ lognormal(0, 0.2);

    for (i in 1:n) {
        real p = a + bt * treatment[i];
        mu[i] = h0[i] * p;
    }

    h1 ~ normal(mu, sigma);
}

generated quantities{
    vector[n] log_probability_density_pointwise;

    for ( i in 1:n ) {
        real p = a + bt * treatment[i];
        real mu = h0[i] * p;
        log_probability_density_pointwise[i] = normal_lpdf(h1[i] | mu, sigma);
    }
}
