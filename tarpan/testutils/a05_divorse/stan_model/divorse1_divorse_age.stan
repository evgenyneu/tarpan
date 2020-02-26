/*
  The model is borrowed from Statistical Rethinking textbook
  by Richard McElreath.
*/

data{
    int n;
    vector[n] D;
    vector[n] A;
}

parameters{
    real a;
    real bA;
    real<lower=0> sigma;
}

model{
    vector[n] mu;
    sigma ~ exponential(1);
    bA ~ normal(0, 0.5);
    a ~ normal(0, 0.2);

    for ( i in 1:n ) {
        mu[i] = a + bA * A[i];
    }

    D ~ normal(mu, sigma);
}

generated quantities{
    vector[n] log_probability_density_pointwise;

    for (i in 1:n) {
        real mu = a + bA * A[i];
        log_probability_density_pointwise[i] = normal_lpdf(D[i] | mu, sigma);
    }
}
