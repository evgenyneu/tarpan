/*
  The model is borrowed from Statistical Rethinking textbook
  by Richard McElreath.
*/

data{
    int n;
    vector[n] D;
    vector[n] A;
    vector[n] M;
}

parameters{
    real a;
    real bM;
    real bA;
    real<lower=0> sigma;
}

model{
    vector[n] mu;
    sigma ~ exponential( 1 );
    bA ~ normal(0, 0.5);
    bM ~ normal(0, 0.5);
    a ~ normal(0, 0.2);

    for (i in 1:n) {
        mu[i] = a + bM * M[i] + bA * A[i];
    }

    D ~ normal( mu , sigma );
}

generated quantities{
    vector[n] log_lik;

    for ( i in 1:n ) {
        real mu = a + bM * M[i] + bA * A[i];
        log_lik[i] = normal_lpdf(D[i] | mu, sigma);
    }
}
