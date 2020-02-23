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
    vector[n] log_lik;
    vector[n] mu;
    for ( i in 1:n ) {
        mu[i] = a + b * speed[i];
    }
    for ( i in 1:n ) log_lik[i] = normal_lpdf( dist[i] | mu[i] , sigma );
}
