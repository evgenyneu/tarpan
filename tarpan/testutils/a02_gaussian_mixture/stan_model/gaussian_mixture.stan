// Guassian mixture model
data {
  int<lower=1> N;        // Number of data points
  real y[N];             // Observed values
  real uncertainties[N]; // Uncertainties
}
parameters {
  real<lower=0,upper=1> r;   // Mixing proportion
  ordered[2] mu;             // Locations of mixture components
  real<lower=0> sigma;       // Spread of mixture components

  // Normally distributed variables
  vector[N] z1;
  vector[N] z2;
}
transformed parameters {
  // Non-centered parametrization
  vector[N] y_mix1 = mu[1] + sigma*z1;
  vector[N] y_mix2 = mu[2] + sigma*z2;
}
model {
  // Set priors
  sigma ~ normal(0, 0.1);
  mu[1] ~ normal(-1, 0.5);
  mu[2] ~ normal(-1, 0.5);

  // Normally distributed variables for non-centered parametrisation
  z1 ~ std_normal();
  z2 ~ std_normal();

  // Loop through observed values
  // and mix two Gaussian components accounting for uncertainty
  // of each measurent
  for (n in 1:N) {
    target += log_mix(1 - r,
                      normal_lpdf(y[n] | y_mix1[n], uncertainties[n]),
                      normal_lpdf(y[n] | y_mix2[n], uncertainties[n]));
  }
}
